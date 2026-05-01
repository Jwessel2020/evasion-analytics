#!/usr/bin/env python3
"""5-fold ensemble inference — average per-head predictions across folds.

For each binary head (speed/search/accident/injury) and each Poisson head
(occurrence/speed_occ/trap), we load that head's best checkpoint from
every fold with an existing `unified_phase2_fold{i}_best_{metric}.pt`,
run inference on the same input data using that fold's train+val-fit
encoders / stats, and average probabilities (binary) or counts (Poisson).

Why per-fold stats: each fold's cat_encoders were built from its own
train+val slice, so indices and normalization differ. Using the wrong
stats explodes test z-scores (see the is_post_covid story).

Why per-head checkpoints: different heads peak at different epochs
(observed: search/accident/injury peak at ep 0-2, speed keeps improving
for 20+ epochs). Using each head's specialist checkpoint beats the
composite-best.

Usage:
    # Score all 1.24M stops + bin grid using every available fold
    python deep/predict_multi_ensemble.py --force

    # Restrict to specific folds (for debugging)
    python deep/predict_multi_ensemble.py --folds 0,1 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.datasets import (
    OccurrenceDataset, StopMultiDataset, load_multi_target_bins,
    CELL_NUMERIC_FEATURES, STOP_CAT_CARDINALITIES, TIME_FEATURES,
)
from src.architecture.unified_model import UnifiedModel


BINARY_HEADS = ("speed", "search", "accident", "injury", "disposition")
POISSON_HEADS = ("occurrence", "speed_occurrence", "trap")

# Map head → best-metric checkpoint suffix
HEAD_TO_CKPT = {
    "speed": "best_speed_auc",
    "search": "best_search_auc",
    "accident": "best_accident_auc",
    "injury": "best_injury_auc",
    "disposition": "best_disposition_auc",  # v3.3.0: new head, per-head ckpt produced by train_multi
    "occurrence": "best_occ_mae",
    "speed_occurrence": "best_speed_occ_mae",
    "trap": "best_trap_mae",
}


def find_available_folds(folds_requested: List[int]) -> List[int]:
    """Return the subset of folds that actually have composite checkpoints."""
    available = []
    for f in folds_requested:
        ckpt = config.MODELS_DIR / f"unified_phase2_fold{f}.pt"
        if ckpt.exists():
            available.append(f)
        else:
            print(f"  fold {f}: {ckpt.name} missing — skipped")
    return available


def _total_stop_cat_dim() -> int:
    """Sum of per-feature embedding dims for STOP_CAT_FEATURES.

    Matches `UnifiedModel.__init__`'s `total_stop_cat_dim = sum(min(c, 32)
    for c in stop_cat_cardinalities)`. Used to back out `n_stop_numeric`
    from a checkpoint's `stop_encoder.0.weight` shape (which is
    `(128, n_stop_numeric + total_stop_cat_dim)`).
    """
    return sum(min(c, 32) for c in STOP_CAT_CARDINALITIES)


def _ckpt_expected_n_stop_numeric(ckpt_path: Path) -> Optional[int]:
    """Peek at a checkpoint's `stop_encoder.0.weight` shape to derive the
    `n_stop_numeric` it was trained with.

    Returns None if the key is missing (shouldn't happen for v3.x but stays
    defensive — caller falls back to dataset-derived count).

    Why we need this: the multi-version landscape forks on whether the
    StopMultiDataset auto-fit TE encoders at training time:
      - v3.0.0 / v3.1.0: pre-runtime-TE → ckpt expects ~112 features
      - v3.1.1+ / v3.2.0 (PLR): runtime-TE → ckpt expects 142 features
    The inference pipeline must build a dataset matching the checkpoint
    or load_state_dict throws a `stop_encoder.0.weight` shape mismatch.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    se_w = state.get("stop_encoder.0.weight")
    if se_w is None:
        return None
    # stop_encoder.0 is Linear(stop_input_dim → 128), so weight shape
    # is (128, stop_input_dim) and n_stop_numeric = stop_input_dim - cat_dim.
    return int(se_w.shape[1]) - _total_stop_cat_dim()


def build_fold_dataset(
    stops_df: pd.DataFrame, bins_df: pd.DataFrame, fold_idx: int, device: str,
    enable_te: bool = True, build_stop: bool = True, build_occ: bool = True,
    leak_safe: bool = False,
) -> Tuple[Optional[StopMultiDataset], Optional[OccurrenceDataset], int]:
    """Rebuild the exact training fold's encoders/stats, then wrap the full
    dataset in those. Guarantees embedding indices match and z-scores don't
    blow up.

    `enable_te` controls whether the StopMultiDataset appends fold-safe
    runtime TE columns. Set False for older checkpoints (v3.0.0/v3.1.0)
    that were trained before runtime TE was integrated into the dataset
    class — those checkpoints expect a smaller `n_stop_numeric` and will
    shape-mismatch otherwise. Defaults to True for the v3.1.1+ era.

    `build_stop` / `build_occ` let callers skip building the dataset they
    don't need. The poisson ensemble only needs occ_ds; the binary
    ensemble only needs stop_ds. Building both eats ~400 MiB of cell_data
    per call (1.24 M stops × 85 features × 4 B), which on the laptop
    A4000 hits the Windows VA limit by fold 3. We still build the
    train_val tds to derive `n_stop_numeric` (cheap — 170 K rows) so the
    caller can validate against the checkpoint shape.
    """
    # v3.3.0: fold-aware citation_rate swap. Mirrors load_fold_split in
    # deep/train_multi.py — fold-N checkpoints were trained with
    # citation_rate = citation_rate_fold{N}; inference must match or
    # z-scores blow up against the wrong stats.
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

    splits = pd.read_parquet(config.PATH_FOLDS)
    fold_col = f"temporal_fold_{fold_idx}"
    joined = stops_df.merge(splits[["stop_id", fold_col]],
                             left_on="id", right_on="stop_id", how="inner")
    train_val = joined[joined[fold_col].isin(["train", "val"])].reset_index(drop=True)

    if enable_te:
        # Fold-safe TE fit: encoders fit on TRAIN ONLY so val (our early-stop
        # signal) doesn't leak into the target encoding. See docs/walk-forward-
        # cv-audit-2026-04-24.md.
        train_only = joined[joined[fold_col] == "train"].reset_index(drop=True)
        from deep.data import _fit_te_encoders, _TE_BINARY_TARGETS
        te_encoders = _fit_te_encoders(train_only, _TE_BINARY_TARGETS)
        tds_stats = {"te_encoders": te_encoders}
    else:
        # No-TE path: pass an empty dict, NOT None. None triggers
        # `_fit_te_encoders` inside StopMultiDataset (the `if stats is not
        # None and "te_encoders" in stats` else branch). An empty dict
        # without the te_encoders key triggers it too. So we have to pass
        # an explicit empty te_encoders dict — _augment_with_te then iterates
        # zero entries and adds zero columns.
        tds_stats = {"te_encoders": {}}

    tds = StopMultiDataset(train_val, stats=tds_stats, leak_safe=leak_safe)
    cat_enc = tds.cat_encoders
    stats = tds.get_stats()
    n_stop_numeric = tds.stop_numeric.shape[1]
    del tds

    stop_ds = (
        StopMultiDataset(stops_df, cat_encoders=cat_enc, stats=stats,
                         leak_safe=leak_safe)
        if build_stop else None
    )
    occ_ds = OccurrenceDataset(bins_df) if build_occ else None
    return stop_ds, occ_ds, n_stop_numeric


def load_ckpt(ckpt_path: Path, n_stop_numeric: int, device: str) -> UnifiedModel:
    """Load a UnifiedModel checkpoint, auto-adapting to v3.x variants.

    Multi-version compat handled here (no CLI flags):
      - Vanilla NumericTokenizer (`weight` + `bias`) vs PLR (`W` +
        `proj_weight` + `proj_bias`) — auto-detected from state_dict keys
        and the backbone's numeric_tokenizer is swapped before loading.
      - `n_stop_numeric` mismatch — caller is responsible for building
        the dataset to match the checkpoint (use
        `_ckpt_expected_n_stop_numeric` to peek). We re-validate here as
        a safety net so a wrong-shape dataset crashes loudly with a clear
        message rather than the cryptic torch shape-mismatch trace.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Safety net: cross-check n_stop_numeric against the checkpoint.
    # The orchestration layer (ensemble_binary/poisson) is supposed to
    # have already inspected the ckpt and built a matching dataset, but
    # if someone calls load_ckpt directly with the wrong number, fail
    # fast with a useful diagnostic.
    se_w = state.get("stop_encoder.0.weight")
    if se_w is not None:
        ckpt_n_stop = int(se_w.shape[1]) - _total_stop_cat_dim()
        if ckpt_n_stop != n_stop_numeric:
            raise RuntimeError(
                f"n_stop_numeric mismatch for {ckpt_path.name}: checkpoint "
                f"expects {ckpt_n_stop} (stop_encoder.0.weight={tuple(se_w.shape)}, "
                f"cat_dim={_total_stop_cat_dim()}), caller built dataset with "
                f"{n_stop_numeric}. Probable cause: pre-runtime-TE checkpoint "
                f"(v3.0.0/v3.1.0 expect ~112) loaded with TE-augmented dataset "
                f"(v3.1.1+/v3.2.0 produce 142), or vice versa. Have orchestration "
                f"call _ckpt_expected_n_stop_numeric() before build_fold_dataset() "
                f"and pass enable_te accordingly."
            )

    model = UnifiedModel(
        n_cell_numeric=len(CELL_NUMERIC_FEATURES),
        cell_cat_cardinalities=[],
        n_time_features=len(TIME_FEATURES),
        n_stop_numeric=n_stop_numeric,
        stop_cat_cardinalities=STOP_CAT_CARDINALITIES,
        d_token=128, n_layers=4, n_heads=8, backbone_dim=128,
    )

    # Auto-detect Phase 3A PLR tokenizer checkpoints (v3.2.0+) and swap.
    # The PLR variant has `W` + `proj_weight` + `proj_bias` keys; vanilla
    # has `weight` + `bias`. Multi-version compat — no flag needed.
    #
    # IMPORTANT: PLR replaces the BACKBONE numeric tokenizer (which sees the
    # CELL numeric features, not the per-stop ones). The checkpoint's
    # `backbone.numeric_tokenizer.W` shape is (n_cell_features, n_periodic_freqs).
    # Pulling n_features from the checkpoint shape (rather than re-deriving
    # from CELL_NUMERIC_FEATURES) keeps us robust to future cell-feature-list
    # drift between training and inference. We assert match below as a
    # safety net.
    if "backbone.numeric_tokenizer.W" in state:
        from next_gen.train import patch_with_plr
        plr_w = state["backbone.numeric_tokenizer.W"]
        n_cell_features_ckpt, n_freqs = plr_w.shape
        if n_cell_features_ckpt != len(CELL_NUMERIC_FEATURES):
            raise RuntimeError(
                f"PLR checkpoint expects {n_cell_features_ckpt} cell numeric "
                f"features but CELL_NUMERIC_FEATURES has "
                f"{len(CELL_NUMERIC_FEATURES)} — feature list drift since "
                f"training. Inference dataset would feed wrong-shape input "
                f"into the PLR tokenizer. Restore the older feature list or "
                f"retrain."
            )
        patch_with_plr(model, n_cell_features_ckpt, d_token=128,
                       n_periodic_freqs=n_freqs)
    model.load_state_dict(state)
    return model.to(device).eval()


def score_binary(
    model: UnifiedModel, stop_ds: StopMultiDataset, device: str,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    n = len(stop_ds)
    out = {h: np.zeros(n, dtype="float32") for h in BINARY_HEADS}
    loader = DataLoader(stop_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pos = 0
    with torch.no_grad():
        for b in loader:
            cn = b["cell_numeric"].to(device)
            cc = b["cell_categorical"].to(device)
            tf = b["time_features"].to(device)
            sn = b["stop_numeric"].to(device)
            sc = b["stop_categorical"].to(device)
            r = model(cn, cc, tf, sn, sc, task="stop_multi")
            for h in BINARY_HEADS:
                # v3.3.0+: 'disposition' head added — older v3.2.0/v3.1.x
                # checkpoints don't have it. Skip if missing (out array
                # stays zeros, which signals "no prediction" downstream).
                if h not in r:
                    continue
                out[h][pos:pos + len(r[h])] = torch.sigmoid(r[h]).cpu().numpy()
            pos += len(r["speed"])
    return out


def score_poisson(
    model: UnifiedModel, occ_ds: OccurrenceDataset, device: str,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    n = len(occ_ds)
    out = {h: np.zeros(n, dtype="float32") for h in POISSON_HEADS}
    loader = DataLoader(occ_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pos = 0
    with torch.no_grad():
        for b in loader:
            cn = b["cell_numeric"].to(device)
            cc = b["cell_categorical"].to(device)
            tf = b["time_features"].to(device)
            r = model(cn, cc, tf, task="occurrence")
            for h in POISSON_HEADS:
                out[h][pos:pos + len(r[h])] = torch.exp(r[h]).cpu().numpy()
            pos += len(r["occurrence"])
    return out


def _decide_te_for_fold(fold_idx: int) -> Tuple[bool, bool, int]:
    """Peek at the fold's composite checkpoint to decide whether the
    inference dataset should append runtime TE columns.

    Returns (enable_te, expected_n_stop_numeric). If the checkpoint's
    expected `n_stop_numeric` matches the TE-augmented dataset size we'd
    produce by default, enable TE. Otherwise (older v3.0.0/v3.1.0
    checkpoints from before runtime TE was integrated), disable it.

    Multi-version compat without a CLI flag: the orchestration uses the
    composite fold checkpoint as the truth source for the fold's
    architecture, since per-head specialist checkpoints are saved with
    the same n_stop_numeric within a fold.
    """
    composite = config.MODELS_DIR / f"unified_phase2_fold{fold_idx}.pt"
    expected = _ckpt_expected_n_stop_numeric(composite)
    # Determine the TE-augmented vs no-TE feature counts the dataset
    # would produce on the current parquet. We can't avoid actually
    # building the dataset to know the no-TE count exactly (the parquet
    # schema may have grown), so we compare against the with-TE count
    # implicitly: if checkpoint expects more than ~28 features fewer than
    # the TE count, disable TE. The threshold of 14 (half a TE-block) is
    # chosen so a small parquet schema drift won't accidentally flip the
    # decision, but a clear "no runtime TE at all" gap (~28 cols) does.
    #
    # In practice the two regimes are well-separated:
    #   - v3.0.0/v3.1.0: ~112 expected
    #   - v3.1.1+/v3.2.0: 142 expected
    # so the threshold is robust.
    if expected is None:
        return True, False, -1  # ckpt unreadable — fall back to default behavior
    # We compute the TE-augmented count once by building the train_only TE,
    # but to avoid that overhead, treat any expected count >= 130 as
    # "TE was active during training" (28-col TE block + ~100 base features
    # is the v3.1.1+ floor).
    enable_te = expected >= 130
    # v3.3.0 leak_safe detection: leak_safe drops is_post_covid +
    # is_covid_era (-2 features). Combined with v3.3.0's smaller TE block
    # (6 cats × 4 = 24 cols vs v3.2.0's 7 cats × 4 = 28), v3.3.0 ckpts
    # land at expected ≈ 136. v3.2.0 ckpts (no leak_safe, full TE) land at 142.
    # Range [130, 137] ⇒ likely v3.3.0 with leak_safe (+/- parquet schema noise).
    leak_safe = 130 <= expected <= 137
    return enable_te, leak_safe, expected


def ensemble_binary(
    stops_df: pd.DataFrame, bins_df: pd.DataFrame,
    folds: List[int], device: str, batch_size: int,
) -> Dict[str, np.ndarray]:
    """For each binary head, average per-stop sigmoid outputs from each
    fold's specialist checkpoint for that head."""
    n = len(stops_df)
    accum = {h: np.zeros(n, dtype="float32") for h in BINARY_HEADS}
    counts = {h: 0 for h in BINARY_HEADS}

    for f in folds:
        enable_te, leak_safe, expected_n = _decide_te_for_fold(f)
        te_str = "with TE" if enable_te else "no TE (legacy ckpt)"
        ls_str = "leak-safe (v3.3.0)" if leak_safe else "full features"
        print(f"\n  fold {f}: building fold-correct dataset ({te_str}, "
              f"{ls_str}, ckpt expects n_stop_numeric={expected_n}) ...", flush=True)
        # Binary path: only the per-stop dataset is needed. Skip occ_ds
        # to save memory (avoids building a 1.14 M-row OccurrenceDataset
        # per fold that the binary loop never reads).
        stop_ds, _, n_stop_numeric = build_fold_dataset(
            stops_df, bins_df, f, device, enable_te=enable_te,
            build_stop=True, build_occ=False, leak_safe=leak_safe,
        )
        for h in BINARY_HEADS:
            ckpt_name = f"unified_phase2_fold{f}_{HEAD_TO_CKPT[h]}.pt"
            ckpt_path = config.MODELS_DIR / ckpt_name
            if not ckpt_path.exists():
                print(f"    {h}: {ckpt_name} missing — falling back to composite")
                ckpt_path = config.MODELS_DIR / f"unified_phase2_fold{f}.pt"
            t0 = time.time()
            model = load_ckpt(ckpt_path, n_stop_numeric, device)
            probs = score_binary(model, stop_ds, device, batch_size)[h]
            dt = time.time() - t0
            print(f"    {h}: mean={probs.mean():.4f} std={probs.std():.4f}  ({dt:.0f}s)",
                  flush=True)
            accum[h] += probs
            counts[h] += 1
            del model
            torch.cuda.empty_cache() if device == "cuda" else None
        del stop_ds
    return {h: accum[h] / max(counts[h], 1) for h in BINARY_HEADS}


def ensemble_poisson(
    stops_df: pd.DataFrame, bins_df: pd.DataFrame,
    folds: List[int], device: str, batch_size: int,
) -> Dict[str, np.ndarray]:
    """For each Poisson head, average per-bin expected counts."""
    n = len(bins_df)
    accum = {h: np.zeros(n, dtype="float32") for h in POISSON_HEADS}
    counts = {h: 0 for h in POISSON_HEADS}

    for f in folds:
        enable_te, leak_safe, expected_n = _decide_te_for_fold(f)
        te_str = "with TE" if enable_te else "no TE (legacy ckpt)"
        ls_str = "leak-safe (v3.3.0)" if leak_safe else "full features"
        print(f"\n  fold {f}: building fold-correct bin dataset ({te_str}, "
              f"{ls_str}, ckpt expects n_stop_numeric={expected_n}) ...", flush=True)
        # Poisson path: only the bin (occurrence) dataset is needed. Skip
        # stop_ds to save the per-fold ~400 MiB cell_data array — this was
        # the OOM trigger when the laptop A4000 hit Windows VA limits at
        # fold 3 of v3.2.0 (the binary phase had already retained some
        # working set, and a fresh 1.24 M-row stop_ds put us over the
        # ~6-8 GB process cap per CLAUDE.md).
        _, occ_ds, n_stop_numeric = build_fold_dataset(
            stops_df, bins_df, f, device, enable_te=enable_te,
            build_stop=False, build_occ=True, leak_safe=leak_safe,
        )
        for h in POISSON_HEADS:
            ckpt_name = f"unified_phase2_fold{f}_{HEAD_TO_CKPT[h]}.pt"
            ckpt_path = config.MODELS_DIR / ckpt_name
            if not ckpt_path.exists():
                ckpt_path = config.MODELS_DIR / f"unified_phase2_fold{f}.pt"
            t0 = time.time()
            model = load_ckpt(ckpt_path, n_stop_numeric, device)
            counts_out = score_poisson(model, occ_ds, device, batch_size)[h]
            dt = time.time() - t0
            print(f"    {h}: mean={counts_out.mean():.4f} ({dt:.0f}s)", flush=True)
            accum[h] += counts_out
            counts[h] += 1
            del model
            torch.cuda.empty_cache() if device == "cuda" else None
    return {h: accum[h] / max(counts[h], 1) for h in POISSON_HEADS}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="0,1,2,3,4",
                   help="Comma-separated fold ids to include")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--force", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-poisson", action="store_true")
    p.add_argument("--skip-binary", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}", flush=True)

    folds_req = [int(x) for x in args.folds.split(",")]
    folds = find_available_folds(folds_req)
    if not folds:
        print("ERROR: no fold checkpoints available")
        return 2
    print(f"Folds in ensemble: {folds}", flush=True)

    stop_out = config.DATA_DIR / "stop_predictions_v4_ensemble.parquet"
    bin_out = config.DATA_DIR / "bin_predictions_v4_ensemble.parquet"
    for p_ in (stop_out, bin_out):
        if p_.exists() and not args.force:
            print(f"ERROR: {p_} exists. Use --force.")
            return 2

    stops = pd.read_parquet(config.PATH_STOPS_FEATURES)
    print(f"stops: {len(stops):,}", flush=True)
    bins = load_multi_target_bins(
        config.DATA_DIR / "occurrence_bins.parquet",
        config.DATA_DIR / "speed_occurrence_bins.parquet",
        config.DATA_DIR / "speed_trap_bins.parquet",
    )
    print(f"bins: {len(bins):,}", flush=True)

    if not args.skip_binary:
        print(f"\n=== BINARY ENSEMBLE ({len(folds)} folds) ===", flush=True)
        bin_probs = ensemble_binary(stops, bins, folds, device, args.batch_size)
        stop_df = pd.DataFrame({
            "stop_id": stops["id"].astype(str).values,
            "is_speed_related_prob": bin_probs["speed"],
            "search_conducted_prob": bin_probs["search"],
            "accident_prob": bin_probs["accident"],
            "personal_injury_prob": bin_probs["injury"],
            # v3.3.0+: disposition prob (P(citation), with warning/SERO as
            # complement). Older v3.2.0/v3.1.x checkpoints don't have this
            # head — ensemble_binary returns zeros which is fine for
            # backward compat (consumers should check for v4.3.0+ tag).
            "disposition_citation_prob": bin_probs.get(
                "disposition", np.zeros(len(stops), dtype="float32")
            ),
        })
        # v3.3.0+: enrich with enforcement metadata for filter UI on /analytics.
        # These columns come straight from stops_features (Phase 7 Layer A in
        # the v3.3.0 plan). Frontend uses them to filter heatmap by marked/
        # unmarked, mobile/stationary, detection method.
        for col in (
            "is_marked_unit", "is_unmarked_unit",
            "is_mobile_enforcement", "is_stationary_trap",
            "is_automated_detection", "detection_method",
            "arrest_type_letter",
        ):
            if col in stops.columns:
                stop_df[col] = stops[col].values
        stop_df.to_parquet(stop_out, compression="zstd", index=False)
        print(f"Wrote {stop_out.name} ({len(stop_df):,} stops, "
              f"{len(stop_df.columns)} cols)", flush=True)

    if not args.skip_poisson:
        print(f"\n=== POISSON ENSEMBLE ({len(folds)} folds) ===", flush=True)
        pois_counts = ensemble_poisson(stops, bins, folds, device, args.batch_size)
        bin_df = bins[["h3_cell", "hour", "day_of_week"]].copy()
        bin_df["occurrence_count"] = pois_counts["occurrence"]
        bin_df["speed_count"] = pois_counts["speed_occurrence"]
        bin_df["trap_count"] = pois_counts["trap"]
        bin_df.to_parquet(bin_out, compression="zstd", index=False)
        print(f"Wrote {bin_out.name} ({len(bin_df):,} bins)", flush=True)

    print("\nEnsemble predictions complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
