#!/usr/bin/env python3
"""End-to-end forward-pass demo of the UnifiedModel architecture.

Run this script after `pip install -e .` to verify the architecture is
correctly wired. No data download, no checkpoint, no GPU required — runs
on CPU in under 5 seconds with synthetic torch.randn inputs.

What it demonstrates:
    1. The UnifiedModel instantiates at production hyperparameters
       (d_token=128, n_layers=4, n_heads=8, ~1.14M trainable params).
    2. The PLR tokenizer + FT-Transformer backbone + 8 task heads
       (3 Poisson + 5 binary) all wire together correctly.
    3. A forward pass on a 16-row batch produces tensors of the expected
       shape for each head — i.e. the multi-task plumbing works.

This is the fastest "is the code real" check for an instructor evaluating
the prototype. The next step (real predictions on real data) requires
either training a checkpoint via `scripts/smoke_train.py` or downloading
one of the v3.3.1 release checkpoints (see README §"Reproducing predictions").

Usage:
    python -m scripts.demo_architecture
    # or, after pip install -e .
    python scripts/demo_architecture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure src/ is importable when run as a plain script (not via -m).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.architecture.unified_model import UnifiedModel
from src.architecture.plr_tokenizer import PLRNumericTokenizer


# Production hyperparameters per the executive report §5.
N_CELL_NUMERIC = 85         # cell-level features (road geom, AADT, POI distances, lens stats, ...)
CELL_CAT_CARDINALITIES = [] # currently no cell-level categoricals
N_TIME_FEATURES = 12        # cyclic encodings (hour, dow, holiday flags, ...)
N_STOP_NUMERIC = 100        # per-stop numeric features
STOP_CAT_CARDINALITIES = [16, 64, 4096, 32, 16, 32]  # 6 categoricals; vehicle_make is the big one
D_TOKEN = 128
N_LAYERS = 4
N_HEADS = 8
BATCH_SIZE = 16


def main() -> int:
    print("=" * 72)
    print("  UnifiedModel architecture demo — synthetic forward pass")
    print("=" * 72)

    # ---- Build model at production hyperparameters ----
    model = UnifiedModel(
        n_cell_numeric=N_CELL_NUMERIC,
        cell_cat_cardinalities=CELL_CAT_CARDINALITIES,
        n_time_features=N_TIME_FEATURES,
        n_stop_numeric=N_STOP_NUMERIC,
        stop_cat_cardinalities=STOP_CAT_CARDINALITIES,
        d_token=D_TOKEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nInstantiated UnifiedModel:")
    print(f"  d_token={D_TOKEN}  n_layers={N_LAYERS}  n_heads={N_HEADS}  k=8 PLR freqs")
    print(f"  trainable params: {n_params:,}")

    # ---- Show PLR tokenizer dimensionality on a sample numeric input ----
    plr = PLRNumericTokenizer(n_features=N_CELL_NUMERIC, d_token=D_TOKEN, n_periodic_freqs=8)
    sample_numeric = torch.randn(BATCH_SIZE, N_CELL_NUMERIC)
    plr_out = plr(sample_numeric)
    print(f"\nPLR tokenizer:")
    print(f"  input shape:  {tuple(sample_numeric.shape)}  (batch x n_numeric)")
    print(f"  output shape: {tuple(plr_out.shape)}  (batch x n_numeric x d_token)")
    print(f"  k=8 frequencies expand each scalar x into "
          f"[x, sin(2pi*W1*x), cos(2pi*W1*x), ..., sin(2pi*W8*x), cos(2pi*W8*x)] "
          f"-> Linear -> ReLU -> {D_TOKEN}-d token")

    # ---- Forward pass with synthetic inputs ----
    cell_numeric = torch.randn(BATCH_SIZE, N_CELL_NUMERIC)
    cell_categorical = torch.zeros(BATCH_SIZE, 0, dtype=torch.long)  # no cell categoricals
    time_features = torch.randn(BATCH_SIZE, N_TIME_FEATURES)
    stop_numeric = torch.randn(BATCH_SIZE, N_STOP_NUMERIC)
    # Per-stop categoricals — must be valid indices for each cardinality.
    stop_categorical = torch.stack(
        [torch.randint(0, c, (BATCH_SIZE,)) for c in STOP_CAT_CARDINALITIES],
        dim=1,
    )

    model.eval()
    with torch.no_grad():
        out = model(
            cell_numeric=cell_numeric,
            cell_categorical=cell_categorical,
            time_features=time_features,
            stop_numeric=stop_numeric,
            stop_categorical=stop_categorical,
            task='both',
        )

    print(f"\nForward pass on a batch of {BATCH_SIZE} synthetic stops:")
    print(f"  3 Poisson heads (cell-hour-day enforcement counts):")
    for h in ("occurrence", "speed_occurrence", "trap"):
        if h in out:
            t = out[h]
            shape_str = "x".join(str(d) for d in t.shape)
            mean = t.exp().mean().item() if t.numel() > 0 else 0.0
            print(f"    {h:18s}  shape=({shape_str:>10s})  exp(log_rate).mean()={mean:.4f}")
    print(f"  5 binary classification heads (per-stop outcomes):")
    for h in ("speed", "search", "accident", "injury", "disposition"):
        if h in out:
            t = out[h]
            shape_str = "x".join(str(d) for d in t.shape)
            mean = torch.sigmoid(t).mean().item() if t.numel() > 0 else 0.0
            print(f"    {h:18s}  shape=({shape_str:>10s})  sigmoid(logit).mean()={mean:.4f}")

    # ---- Sanity check: all 8 heads produced the right number of outputs ----
    expected = {"occurrence", "speed_occurrence", "trap",
                "speed", "search", "accident", "injury", "disposition"}
    got = set(out.keys())
    missing = expected - got
    if missing:
        print(f"\nERROR: missing heads in forward output: {missing}")
        return 2

    print("\n" + "=" * 72)
    print("  All 8 heads produced output. Architecture is wired correctly.")
    print("  Next: run scripts/smoke_train.py for a 2-epoch training loop on")
    print("  the sample dataset, or download a v3.3.1 release checkpoint and")
    print("  use scripts/smoke_inference.py for predictions on real stops.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
