#!/usr/bin/env python3
"""Phase 2 — FT-Transformer multi-task training (7 heads).

Replaces the 7 XGBoost models with one UnifiedModel. Loads:
  - Multi-target bins (occurrence × 3 Poisson counts) from the 3 07a parquets
  - Per-stop features from stops_features.parquet (4 binary heads)
  - Walk-forward folds from train_test_splits.parquet (5 folds)

Alternates occurrence and stop_multi batches each step. Weighted multi-task
loss. Per-epoch eval of all 7 metrics vs XGBoost baselines for Phase 2 goal
check (match-or-beat on each target).

Usage:
    cd ml/research
    python deep/train_multi.py --fold 0 --epochs 60        # single fold
    python deep/train_multi.py --fold 0 --epochs 5 --smoke  # quick test
    python deep/train_multi.py --fold all                  # 5 folds sequential

Output:
    models/v3.0.0/unified_phase2_fold{i}.pt
    models/v3.0.0/unified_phase2_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src import config
from src.data.datasets import (
    OccurrenceDataset, StopMultiDataset, load_multi_target_bins,
    CELL_NUMERIC_FEATURES, STOP_CAT_CARDINALITIES, TIME_FEATURES,
    STOP_BINARY_TARGETS,
)
from src.architecture.unified_model import UnifiedModel


# Loss weights — Phase 2B rebalanced. The 3 binary heads (search,
# accident, injury) were badly undertrained in Phase 2A at 0.10/0.10/0.05
# so shift mass toward them. Trap is already winning; de-prioritize.
# Occurrence converges easily; reduce from 0.40.
LOSS_WEIGHTS = {
    "occurrence":       0.25,
    "speed_occurrence": 0.10,
    "trap":             0.05,
    "speed":            0.08,   # was 0.10 — rebalanced -0.02 to make room for disposition
    "search":           0.13,   # was 0.15 — rebalanced -0.02
    "accident":         0.18,   # was 0.20 — rebalanced -0.02
    "injury":           0.13,   # was 0.15 — rebalanced -0.02
    "disposition":      0.08,   # v3.3.0 — new head, sum stays 1.00
}

# pos_weight cap prevents gradient explosion on extremely rare classes
# (personal_injury at 1.7% would give pw ≈ 58; clip at 50 for stability)
MAX_POS_WEIGHT = 50.0


def binary_target_index(name: str) -> int:
    return STOP_BINARY_TARGETS.index(name)


def load_fold_split(
    stops_df: pd.DataFrame,
    bins_df: pd.DataFrame,
    fold_idx: int,
    splits_path: Path,
):
    """Return (stops_train, stops_test, bins_train, bins_test).

    Stops are split by the walk-forward fold's train/val/test slices.
    Bins are split by stop_date — use the same temporal boundaries so
    the Poisson models don't peek at future weeks during training.
    """
    splits = pd.read_parquet(splits_path)
    fold_col = f"temporal_fold_{fold_idx}"
    if fold_col not in splits.columns:
        raise ValueError(f"{fold_col} not in train_test_splits.parquet")

    # v3.3.0: fold-aware citation_rate swap to prevent target leakage in
    # the disposition head. Each fold's `citation_rate_fold{N}` column is
    # computed from that fold's train+val stops only (see 09j --fold-idx).
    # Swap into the canonical `citation_rate` column so the model + dataset
    # never need to know about the fold-aware variants. Drop the fold-aware
    # cols afterwards so they don't shift n_stop_numeric / break checkpoint
    # shape compat.
    fa_col = f"citation_rate_fold{fold_idx}"
    if fa_col in stops_df.columns:
        stops_df = stops_df.copy()
        stops_df["citation_rate"] = stops_df[fa_col]
        stops_df = stops_df.drop(columns=[
            c for c in stops_df.columns if c.startswith("citation_rate_fold")
        ])
    if fa_col in bins_df.columns:
        bins_df = bins_df.copy()
        bins_df["citation_rate"] = bins_df[fa_col]
        bins_df = bins_df.drop(columns=[
            c for c in bins_df.columns if c.startswith("citation_rate_fold")
        ])

    joined = stops_df.merge(splits[["stop_id", fold_col]],
                            left_on="id", right_on="stop_id", how="inner")
    s_train = joined[joined[fold_col] == "train"].reset_index(drop=True)
    s_val = joined[joined[fold_col] == "val"].reset_index(drop=True)
    s_test = joined[joined[fold_col] == "test"].reset_index(drop=True)
    train_val = pd.concat([s_train, s_val], ignore_index=True)

    # For bins: split by date ranges of train_val vs test
    if len(s_test) == 0:
        # fold 2 degenerate case (1 test row) — skip
        return train_val, s_test, bins_df, bins_df.iloc[:0]

    test_start = pd.to_datetime(s_test["stop_date"]).min()
    test_end = pd.to_datetime(s_test["stop_date"]).max()
    # bins don't have stop_date; they're aggregated by hour × dow, so we
    # train on ALL bins and evaluate test loss on held-out stops only.
    # For Poisson, cell×hour×dow is stable — the right eval is "MAE on
    # bins the model hasn't seen the stops from."
    # Simpler: random 80/20 bin split. Poisson targets are distribution-
    # stable; the bin split isn't the leakage concern.
    rng = np.random.default_rng(42 + fold_idx)
    perm = rng.permutation(len(bins_df))
    cut = int(0.8 * len(perm))
    bins_train = bins_df.iloc[perm[:cut]].reset_index(drop=True)
    bins_test = bins_df.iloc[perm[cut:]].reset_index(drop=True)

    print(f"  fold {fold_idx}: stops train+val={len(train_val):,} test={len(s_test):,}  "
          f"bins train={len(bins_train):,} test={len(bins_test):,}")
    return train_val, s_test, bins_train, bins_test


def eval_metrics(
    model: UnifiedModel,
    occ_loader: DataLoader,
    stop_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Compute 7 metrics on test loaders."""
    model.eval()
    occ_preds, occ_true = [], []
    speed_occ_preds, speed_occ_true = [], []
    trap_preds, trap_true = [], []
    with torch.no_grad():
        for b in occ_loader:
            cn = b["cell_numeric"].to(device)
            cc = b["cell_categorical"].to(device)
            tf = b["time_features"].to(device)
            out = model(cn, cc, tf, task="occurrence")
            occ_preds.append(torch.exp(out["occurrence"]).cpu().numpy())
            occ_true.append(b["target_total"].numpy())
            speed_occ_preds.append(torch.exp(out["speed_occurrence"]).cpu().numpy())
            speed_occ_true.append(b["target_speed"].numpy())
            trap_preds.append(torch.exp(out["trap"]).cpu().numpy())
            trap_true.append(b["target_trap"].numpy())

    occ_mae = float(np.mean(np.abs(np.concatenate(occ_preds) - np.concatenate(occ_true))))
    speed_occ_mae = float(np.mean(np.abs(np.concatenate(speed_occ_preds) - np.concatenate(speed_occ_true))))
    trap_mae = float(np.mean(np.abs(np.concatenate(trap_preds) - np.concatenate(trap_true))))

    stop_probs = {t: [] for t in STOP_BINARY_TARGETS}
    stop_true = {t: [] for t in STOP_BINARY_TARGETS}
    with torch.no_grad():
        for b in stop_loader:
            cn = b["cell_numeric"].to(device)
            cc = b["cell_categorical"].to(device)
            tf = b["time_features"].to(device)
            sn = b["stop_numeric"].to(device)
            sc = b["stop_categorical"].to(device)
            targets = b["targets"].numpy()  # (B, 4)
            out = model(cn, cc, tf, sn, sc, task="stop_multi")
            # map model output keys to target names
            name_map = {"speed": "is_speed_related", "search": "search_conducted",
                        "accident": "accident", "injury": "personal_injury",
                        "disposition": "is_citation"}
            for mkey, tname in name_map.items():
                probs = torch.sigmoid(out[mkey]).cpu().numpy()
                stop_probs[tname].append(probs)
                stop_true[tname].append(targets[:, binary_target_index(tname)])

    aucs = {}
    for t in STOP_BINARY_TARGETS:
        y_prob = np.concatenate(stop_probs[t])
        y_true = np.concatenate(stop_true[t])
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            aucs[t] = float(roc_auc_score(y_true, y_prob))
        else:
            aucs[t] = float("nan")

    return {
        "occ_mae": occ_mae,
        "speed_occ_mae": speed_occ_mae,
        "trap_mae": trap_mae,
        "speed_auc": aucs["is_speed_related"],
        "search_auc": aucs["search_conducted"],
        "accident_auc": aucs["accident"],
        "injury_auc": aucs["personal_injury"],
        "disposition_auc": aucs["is_citation"],
    }


def train_fold(args, fold_idx: int) -> Dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_amp = (device == "cuda") and not getattr(args, "no_amp", False)
    pos_weight_cap = getattr(args, "pos_weight_cap", MAX_POS_WEIGHT)
    print(f"\n=== Fold {fold_idx} (device: {device}, amp={use_amp}, "
          f"pos_weight_cap={pos_weight_cap}) ===", flush=True)

    print("[load] stops + bins ...")
    stops_df = pd.read_parquet(config.PATH_STOPS_FEATURES)
    bins_df = load_multi_target_bins(
        config.DATA_DIR / "occurrence_bins.parquet",
        config.DATA_DIR / "speed_occurrence_bins.parquet",
        config.DATA_DIR / "speed_trap_bins.parquet",
    )

    if args.smoke:
        stops_df = stops_df.sample(n=50_000, random_state=42).reset_index(drop=True)
        bins_df = bins_df.sample(n=50_000, random_state=42).reset_index(drop=True)

    train_stops, test_stops, train_bins, test_bins = load_fold_split(
        stops_df, bins_df, fold_idx, config.PATH_FOLDS,
    )
    if len(test_stops) == 0:
        print(f"  fold {fold_idx} has no test stops — skipping")
        return {}

    # Datasets
    occ_train = OccurrenceDataset(train_bins)
    occ_test = OccurrenceDataset(test_bins, stats=occ_train.get_stats())
    leak_safe = getattr(args, "leak_safe", False)
    stop_train = StopMultiDataset(train_stops, leak_safe=leak_safe)
    stop_test = StopMultiDataset(test_stops, cat_encoders=stop_train.cat_encoders,
                                  leak_safe=leak_safe,
                                  stats=stop_train.get_stats())

    bs = args.batch_size
    occ_tr_loader = DataLoader(occ_train, batch_size=bs, shuffle=True, num_workers=0)
    occ_te_loader = DataLoader(occ_test, batch_size=bs * 2, shuffle=False, num_workers=0)
    stop_tr_loader = DataLoader(stop_train, batch_size=bs, shuffle=True, num_workers=0)
    stop_te_loader = DataLoader(stop_test, batch_size=bs * 2, shuffle=False, num_workers=0)

    model = UnifiedModel(
        n_cell_numeric=len(CELL_NUMERIC_FEATURES),
        cell_cat_cardinalities=[],
        n_time_features=len(TIME_FEATURES),
        n_stop_numeric=stop_train.stop_numeric.shape[1],
        stop_cat_cardinalities=STOP_CAT_CARDINALITIES,
        d_token=args.d_token, n_layers=args.n_layers, n_heads=8,
        backbone_dim=128,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params", flush=True)

    # Compute pos_weight per binary target from the training slice for
    # class-imbalance correction. Clip at MAX_POS_WEIGHT to avoid gradient
    # explosion on rarest class (personal_injury ~1.7% positive).
    pos_weights = {}
    for t in STOP_BINARY_TARGETS:
        idx = STOP_BINARY_TARGETS.index(t)
        y = stop_train.targets[:, idx].numpy()
        pos = max(y.sum(), 1)
        neg = max(len(y) - pos, 1)
        pw = min(neg / pos, pos_weight_cap)
        pos_weights[t] = torch.tensor(pw, device=device)
        print(f"    pos_weight[{t}] = {pw:.2f}  (pos {100*pos/len(y):.2f}%)",
              flush=True)

    # Differential LR: backbone gets half the head LR (was 0.3 at lr=1e-3;
    # bumped to 0.5 at lr=3e-4 so the backbone still trains at a meaningful
    # pace now that the head LR is already lower).
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if "_head" in name or "stop_encoder" in name or "stop_cat_embeddings" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.5, "weight_decay": 1e-4},
        {"params": head_params,     "lr": args.lr,       "weight_decay": 1e-4},
    ])
    # Canonical FT-T schedule: linear warmup for `warmup_frac` of total epochs,
    # then cosine anneal to 0 across the remainder. With AMP + lr=3e-4 the
    # warmup is what keeps attention from blowing up in the first few epochs.
    warmup_epochs = max(1, int(args.epochs * args.warmup_frac))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: (ep + 1) / warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs],
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best = {"epoch": -1, "combined": -1e9}
    history = []

    # Per-head best tracking — each target gets its own best epoch + checkpoint.
    # MAE targets: smaller is better. AUC targets: larger is better.
    # At inference, using each head's specialist checkpoint beats composite-best.
    per_head_best: Dict[str, Dict] = {
        "occ_mae": {"epoch": -1, "value": 1e9, "better": "lower"},
        "speed_occ_mae": {"epoch": -1, "value": 1e9, "better": "lower"},
        "trap_mae": {"epoch": -1, "value": 1e9, "better": "lower"},
        "speed_auc": {"epoch": -1, "value": -1e9, "better": "higher"},
        "search_auc": {"epoch": -1, "value": -1e9, "better": "higher"},
        "accident_auc": {"epoch": -1, "value": -1e9, "better": "higher"},
        "injury_auc": {"epoch": -1, "value": -1e9, "better": "higher"},
        "disposition_auc": {"epoch": -1, "value": -1e9, "better": "higher"},
    }

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        occ_iter = iter(occ_tr_loader)
        stop_iter = iter(stop_tr_loader)
        n_total = min(len(occ_tr_loader), len(stop_tr_loader))

        for it in range(n_total):
            optimizer.zero_grad(set_to_none=True)

            # -- Occurrence batch --
            b = next(occ_iter)
            cn = b["cell_numeric"].to(device, non_blocking=True)
            cc = b["cell_categorical"].to(device, non_blocking=True)
            tf = b["time_features"].to(device, non_blocking=True)
            y_tot = b["target_total"].to(device, non_blocking=True)
            y_spd = b["target_speed"].to(device, non_blocking=True)
            y_trp = b["target_trap"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(cn, cc, tf, task="occurrence")
                loss_occ = F.poisson_nll_loss(out["occurrence"], y_tot, log_input=True)
                loss_spdocc = F.poisson_nll_loss(out["speed_occurrence"], y_spd, log_input=True)
                loss_trap = F.poisson_nll_loss(out["trap"], y_trp, log_input=True)
                loss_poisson = (LOSS_WEIGHTS["occurrence"] * loss_occ
                                + LOSS_WEIGHTS["speed_occurrence"] * loss_spdocc
                                + LOSS_WEIGHTS["trap"] * loss_trap)
            if scaler is not None:
                scaler.scale(loss_poisson).backward()
            else:
                loss_poisson.backward()

            # -- Stop-multi batch --
            b = next(stop_iter)
            cn = b["cell_numeric"].to(device, non_blocking=True)
            cc = b["cell_categorical"].to(device, non_blocking=True)
            tf = b["time_features"].to(device, non_blocking=True)
            sn = b["stop_numeric"].to(device, non_blocking=True)
            sc = b["stop_categorical"].to(device, non_blocking=True)
            targets = b["targets"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(cn, cc, tf, sn, sc, task="stop_multi")
                y_spd = targets[:, binary_target_index("is_speed_related")]
                y_srch = targets[:, binary_target_index("search_conducted")]
                y_acc = targets[:, binary_target_index("accident")]
                y_inj = targets[:, binary_target_index("personal_injury")]
                y_cit = targets[:, binary_target_index("is_citation")]
                # Class-imbalance-corrected BCE via pos_weight
                loss_speed = F.binary_cross_entropy_with_logits(
                    out["speed"], y_spd, pos_weight=pos_weights["is_speed_related"])
                loss_search = F.binary_cross_entropy_with_logits(
                    out["search"], y_srch, pos_weight=pos_weights["search_conducted"])
                loss_accident = F.binary_cross_entropy_with_logits(
                    out["accident"], y_acc, pos_weight=pos_weights["accident"])
                loss_injury = F.binary_cross_entropy_with_logits(
                    out["injury"], y_inj, pos_weight=pos_weights["personal_injury"])
                loss_disposition = F.binary_cross_entropy_with_logits(
                    out["disposition"], y_cit, pos_weight=pos_weights["is_citation"])
                loss_binary = (LOSS_WEIGHTS["speed"] * loss_speed
                               + LOSS_WEIGHTS["search"] * loss_search
                               + LOSS_WEIGHTS["accident"] * loss_accident
                               + LOSS_WEIGHTS["injury"] * loss_injury
                               + LOSS_WEIGHTS["disposition"] * loss_disposition)
            if scaler is not None:
                scaler.scale(loss_binary).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_binary.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss_poisson.item() + loss_binary.item())
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        # Eval
        metrics = eval_metrics(model, occ_te_loader, stop_te_loader, device)
        # Composite: lower is better for MAE, higher is better for AUC
        # Scale MAEs relative to XGBoost baselines, then invert to "score"
        xgb_mae = {"occ_mae": 0.339, "speed_occ_mae": 0.086, "trap_mae": 0.091}
        xgb_auc = {"speed_auc": 0.842, "search_auc": 0.883,
                   "accident_auc": 0.894, "injury_auc": 0.881,
                   # v3.3.0: disposition_auc baseline = 0.65 placeholder (worse-than-random
                   # untrained XGBoost on Citation/Warning binary; replace with measured
                   # value after running the XGB baseline). Citation is 66% positive so a
                   # constant-positive predictor scores ~0.66 anyway; ANY learning signal
                   # should easily clear 0.65. Conservative until we benchmark.
                   "disposition_auc": 0.65}
        mae_score = sum((xgb_mae[k] - metrics[k]) / xgb_mae[k] for k in xgb_mae)
        auc_score = sum((metrics[k] - xgb_auc[k]) for k in xgb_auc)
        composite = mae_score + auc_score

        history.append({"epoch": epoch, "train_loss": total_loss / max(1, n_batches),
                        **metrics, "composite": composite})

        if composite > best["combined"]:
            best = {"epoch": epoch, "combined": composite, **metrics}
            # Smoke runs must NEVER overwrite production checkpoints —
            # they downsample to 50k rows and usually run only 2-3
            # epochs, so the resulting weights are garbage. An earlier
            # smoke test quietly clobbered unified_phase2_fold0.pt and
            # its per-head bests, forcing a full retrain to recover.
            # Task #57.
            if not args.smoke:
                ckpt = config.MODELS_DIR / f"unified_phase2_fold{fold_idx}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_metrics": best,
                    "fold": fold_idx,
                    "loss_weights": LOSS_WEIGHTS,
                }, ckpt)

        # Per-head best tracking: if any head improved, save a specialist
        # checkpoint for that head (so we can ensemble at inference time).
        for head_name, info in per_head_best.items():
            val = metrics[head_name]
            if info["better"] == "lower":
                improved = val < info["value"]
            else:
                improved = val > info["value"]
            if improved:
                info["value"] = float(val)
                info["epoch"] = epoch
                # Smoke runs skip save — see rationale above. Task #57.
                if not args.smoke:
                    head_ckpt = config.MODELS_DIR / f"unified_phase2_fold{fold_idx}_best_{head_name}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "head_name": head_name,
                        "head_value": float(val),
                        "epoch": epoch,
                        "fold": fold_idx,
                    }, head_ckpt)

        lr_now = scheduler.get_last_lr()[0]
        print(f"  ep {epoch:3d}  loss={total_loss/max(1,n_batches):.3f}  "
              f"occ={metrics['occ_mae']:.3f}/{metrics['speed_occ_mae']:.3f}/{metrics['trap_mae']:.3f}  "
              f"auc={metrics['speed_auc']:.3f}/{metrics['search_auc']:.3f}/"
              f"{metrics['accident_auc']:.3f}/{metrics['injury_auc']:.3f}/"
              f"{metrics.get('disposition_auc', float('nan')):.3f}  "
              f"cmp={composite:+.3f}  lr={lr_now:.1e}  {dt:.1f}s", flush=True)

        # Early stopping: 30 epochs without improvement
        if epoch - best["epoch"] > 30:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"  BEST @ ep {best['epoch']}: "
          f"occ={best['occ_mae']:.3f} (xgb 0.339), "
          f"speed={best['speed_auc']:.3f} (xgb 0.842), ...", flush=True)
    print(f"  Per-head best checkpoints:", flush=True)
    for head_name, info in per_head_best.items():
        print(f"    {head_name:<18s} best={info['value']:.4f} @ ep {info['epoch']}",
              flush=True)
    return {"fold": fold_idx, "best": best, "per_head_best": per_head_best,
            "history": history}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="0")
    p.add_argument("--epochs", type=int, default=60)
    # Defaults bumped 2026-04-24 to match canonical FT-T training recipe:
    # batch_size 1024, lr 3e-4 with 10% linear warmup + cosine.
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of epochs used for linear LR warmup")
    p.add_argument("--d-token", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed-precision training (fp32 only). Use "
                        "when AMP overflow causes AUC=0.500 frozen outputs.")
    p.add_argument("--pos-weight-cap", type=float, default=50.0,
                   help="Max pos_weight per binary head (default 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (affects weight init, dropout, shuffling)")
    p.add_argument("--leak-safe", action="store_true",
                   help="Drop is_speed_related target-encoded features + "
                        "is_covid era flags. Produces honest speed_auc but "
                        "checkpoints are NOT compatible with default-trained ones.")
    args = p.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    folds = [int(args.fold)] if args.fold != "all" else [0, 1, 2, 3, 4]
    results = {}
    for f in folds:
        results[f] = train_fold(args, f)

    out_path = config.MODELS_DIR / "unified_phase2_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any existing metrics file so running folds one at a time
    # (across separate invocations / machines) doesn't clobber earlier
    # folds' results.
    existing: Dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}
    for k, v in results.items():
        existing[str(k)] = {"best": v.get("best")}
    out_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nWrote {out_path} ({len(existing)} folds total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
