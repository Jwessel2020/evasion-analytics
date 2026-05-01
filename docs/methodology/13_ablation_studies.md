# Ablation Studies

**Drafted 2026-04-25.** What we've measured (almost nothing) and what we should measure. The 8-cell Phase 3 grid + the slim-grid strategy that keeps total compute under control.

---

## Purpose

Without ablation, every architectural choice is a guess. v3.1.0 has dozens of choices (d_token=64, n_layers=3, n_heads=4, attention dropout=0.2, ReGLU FFN, residual dropout=0.1, differential LR, alternating batches, loss-weight pattern, pos_weight cap=50, ...) — almost none of which were ablated. Phase 3 adds 5 more components (PLR + MoE + decorrelation + head-attention + GAT). Without a slim grid, we won't know what's actually doing the work.

---

## Current state (v3.1.0)

### What HAS been ablated

| Ablation | Where | Finding |
|---|---|---|
| Road + POI feature removal | `pipelines/ablation_road_poi.py` | XGBoost-baseline only; impact on speed AUC measured |
| 2026-04-24 loss rebalance | `train_multi.py` | Empirical (one-shot, not grid) — boosted occurrence weight from 0.10 to 0.25 |
| Phase 0C label partition | `07a_speed_occurrence.py` (mobile vs stationary) | Speed/trap correlation 0.95 → 0.054 |
| Phase 0D vehicle-targeting log-ratios | `09g_precompute_driver_features.py` | Added; impact on composite measured indirectly via v3.1.0 vs v3.0.0 |
| Phase 3A PLR (smoke fold 4, 30 ep, 2026-04-25) | `next_gen/tokenizers/plr.py` | Smoke +0.150 composite delta; gated to ship |
| Phase 3A PLR (smoke fold 0, 30 ep, 2026-04-25) | same | +0.360 vs vanilla — bigger lift on the HARD fold |
| Phase 3D decorrelation (smoke fold 4, 2026-04-25) | `next_gen/losses/decorrelation.py` | NO-OP — Phase 0C upstream made it nothing-to-correct |
| Phase 3A PLR full retrain (5-fold, 2026-04-26) | `next_gen/train.py --use-plr` (60 ep × 4 folds) | fold 0 +0.292, fold 1 +0.657, fold 3 (just done), fold 4 **+1.143**; mean ~+0.70 vs v3.1.0 +0.29 |

### Phase 3 component status snapshot (post-v3.2.0 + v3.3.0 build)

| Component | Status | Latest result | Ship decision |
|---|---|---|---|
| 3A PLR | ✅ **SHIPPED in v3.2.0** (deployed `v4.2.0-ensemble`, 331K rows) | mean +0.775 across 4 folds | Shipped 2026-04-26 |
| 3B MoE Poisson heads | ❌ **smoke FAILED** 2026-04-26 | fold 4 30ep: comp +0.917 (Δ -0.226 vs +1.143) | DROPPED for v3.3.0; branch `next-gen/p4-moe-heads` preserved |
| 3C Head-attention masks | ⏳ not built | — | Deprioritized — PLR captured most expected lift |
| 3D Decorrelation loss | ❌ NO-OP | smoke confirmed nothing to correct | Will not ship |
| 3E Spatial GAT | ❌ **smoke FAILED** 2026-04-26 | fold 0 30ep: comp +0.136 (Δ -0.156 vs +0.292) | DROPPED for v3.3.0; branch `next-gen/p2-spatial-cache` preserved |
| **v3.3.0 disposition head** (NEW) | ✅ **smoke PASSED** 2026-04-26 evening | fold 0 60ep: comp +0.336 (Δ +0.044 vs +0.292) | Shipping in v3.3.0 = PLR + disposition |

### Phase 2 component status snapshot

| Component | Status | Latest result | Ship decision |
|---|---|---|---|
| 2 LSTM `temporal_fuse` | ❌ **smoke FAILED** 2026-04-26 | comp Δ -0.067 vs PLR baseline | DROPPED; revisit if cell sequence quality improves |
| Hawkes intensity head | ❌ **smoke FAILED** 2026-04-26 | fold 0 60ep: comp +0.262 / occ_mae 0.395 (Δ -0.030 cmp / +0.003 occ_mae vs gates) | DROPPED for v3.3.0; branch `next-gen/p5-hawkes` preserved |

### What HAS NOT been ablated

| Knob | Reason it matters | Cost estimate |
|---|---|---|
| `d_token` (32 / 64 / 96 / 128) | Tokenizer capacity | 4 folds × 4 settings × 60 epochs ≈ 80 GPU-hrs |
| `n_layers` (2 / 3 / 4) | Backbone depth | 4 × 3 × 60 ≈ 60 GPU-hrs |
| `n_heads` (2 / 4 / 8) | Multi-head capacity | 60 GPU-hrs |
| FFN type (ReGLU / SwiGLU / GeLU) | FFN nonlinearity | 60 GPU-hrs |
| Pre-norm vs post-norm | Stability | 40 GPU-hrs |
| Attention dropout (0.0 / 0.1 / 0.2 / 0.3) | Regularization | 80 GPU-hrs |
| Residual dropout (0.0 / 0.1 / 0.2) | Regularization | 60 GPU-hrs |
| LR ratio (backbone × 0.25 / 0.5 / 1.0 / 2.0) | Differential LR | 80 GPU-hrs |
| Loss weights (9 patterns) | Multi-task balance | 180 GPU-hrs |
| pos_weight cap (30 / 50 / 75 / 100) | Imbalance handling | 80 GPU-hrs |
| Per-head MLP depth (1 / 2 / 3 layers) | Head capacity | 60 GPU-hrs |
| Phase 2 LSTM on / off | Temporal lift | 40 GPU-hrs |
| Phase 3 GAT on / off | Spatial lift | 40 GPU-hrs |
| `--smoke` cap (5 / 10 / 20 epochs) | Smoke-test fidelity | 30 GPU-hrs |
| Scheduler (linear / cosine / step) | LR schedule | 60 GPU-hrs |
| Optimizer (AdamW / Adan / Lion) | Optimizer choice | 60 GPU-hrs |
| AMP fp16 vs fp32 | Numerical precision | 40 GPU-hrs |
| Batch size (16 / 32 / 64) | Effective batch | 60 GPU-hrs |
| Embargo length (60 / 90 / 120 days) | CV honesty | 60 GPU-hrs |

**If we did everything: ~1100+ GPU-hrs**. Cost prohibitive at $0.85/hr Modal × 5 folds = $5K. Local compute (laptop A4000 + desktop 4070 SUPER) is "free" but takes weeks of wall time.

---

## Known gaps / pain points

- **No automated ablation pipeline**. Each variant requires manual `python train_multi.py --fold X --epochs Y`. No grid runner.
- **No early-stop ablation strategy**. Could shortcut to "AUC at epoch 30" instead of "best across 60 epochs" — saves ~50% wall time per cell.
- **Composite metric obscures per-head wins**. An ablation that helps speed by 0.02 AUC but hurts injury by 0.01 may net to zero composite. Need per-head attribution.
- **No baseline drift monitoring**. If the data changes (e.g., recovery sweep adds 31 features), all old ablation results are stale by definition.
- **No statistical significance testing**. Single-point AUC numbers don't tell us if 0.842 vs 0.843 is real.
- **Ablations conflict with production retrains**. Same GPU; ablation grids block weekly model refresh.

---

## Open questions

- Should we run full ablation grid on Modal ($) or accept slow wall-time on local (free)?
- Should ablations use `--smoke` 5 epochs (fast, noisy) or full 60 epochs (slow, clean)?
- Should we ablate ONLY on fold 4 (best, most data) or use mean across folds?
- Should we adopt "Pareto dominates" definition: variant V dominates V' iff V wins or ties on every head?

---

## Next-gen direction — Phase 3 slim grid

For Phase 3 component validation, do NOT do a full grid. Do a slim 3-axis × 8-cell grid:

```
                 PLR=off, PLR=on
              ┌─────────┬─────────┐
MoE=off,  ┌── │  cell 1 │ cell 2  │  ← decorr=off
GAT=off,  │   ├─────────┼─────────┤
          └── │  cell 3 │ cell 4  │  ← decorr=on
              └─────────┴─────────┘
              ┌─────────┬─────────┐
MoE=on,   ┌── │  cell 5 │ cell 6  │  ← decorr=off
GAT=on,   │   ├─────────┼─────────┤
          └── │  cell 7 │ cell 8  │  ← decorr=on
              └─────────┴─────────┘
```

8 cells × fold 4 only × 30 epochs ≈ 16 GPU-hrs ≈ $14 on Modal. Acceptable.

**Per-cell metrics**: composite + per-head MAE/AUC + CKA(speed_head, trap_head) + attention sparsity (per-head L1 of attention masks).

**Decision rule**: ship the cell with HIGHEST composite + CKA < 0.70 + at least 2 head AUCs above v3.1.0 baseline. If no cell meets gates, ablate per-component (which one dragged) and revert.

### Other proposed ablations (post-Phase-3)

| Priority | Ablation | Cost | Why |
|---|---|---|---|
| HIGH | `d_token` ∈ {64, 96, 128} | 12 GPU-hrs | Likely too small for 247 features |
| HIGH | Loss-weight grid (9 patterns) | 30 GPU-hrs | Currently set empirically once |
| MED | Per-head MLP depth (1 / 2 / 3) | 12 GPU-hrs | Too shallow may bottleneck |
| MED | Phase 2 LSTM on/off | 8 GPU-hrs | Currently always on; cost unmeasured |
| LOW | FFN type (ReGLU vs SwiGLU) | 8 GPU-hrs | Likely ≤0.005 AUC delta |
| LOW | Optimizer (AdamW / Adan / Lion) | 12 GPU-hrs | AdamW probably fine |

---

## Smoke-grid — "cheap probe" pattern

For any ablation, run smoke (5 epochs, 1 fold, ~10 min) FIRST to filter blatantly bad settings. Then run full (60 epochs, 1 fold, ~2 hr) on top 2-3 candidates from smoke.

Saves ~70% compute by killing bad cells early.

---

## References

- [`05_model_architecture.md`](05_model_architecture.md) — what each architectural knob means
- [`06_hyperparameters.md`](06_hyperparameters.md) — current values
- [`09_evaluation_metrics.md`](09_evaluation_metrics.md) — what to measure per ablation cell
- `docs/research/feature-engineering-catalog-2026-04-24.md` — feature ablation candidates
- `archive/old-reports/research-v3-stages-01-06-report.md` — v3 stage-by-stage progression

---

## TODO

- [ ] Build `scripts/run_ablation_grid.py` (config-driven, runs N variants sequentially, writes `ablation_results.json`)
- [ ] Phase 3 slim 8-cell grid on fold 4
- [ ] Smoke-grid pattern (5 epochs filter → full 60 on top candidates)
- [ ] `d_token` sweep (highest priority post-Phase-3)
- [ ] Loss-weight grid sweep (9 patterns, fold 4 only)
- [ ] Statistical significance: bootstrap on per-cell composite (1000 iter)
- [ ] Pareto-dominance definition + reporting in ablation results
