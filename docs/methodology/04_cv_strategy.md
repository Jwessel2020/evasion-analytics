# Cross-Validation Strategy

**Drafted 2026-04-25.** Walk-forward 5-fold + 90-day embargo + train-only TE encoder fits. Why this matters for production-grounded metrics, what fold 2 degeneracy means, and where the strategy still has gaps.

---

## Purpose

Random K-fold on time-series data gives optimistic AUCs because the model learns from "future" data when predicting "past." Walk-forward forces the model to predict only forward-in-time, matching what production does.

The 2026-04-23 speed-AUC = 1.000 leak (`docs/audits/speed-auc-leak-investigation-2026-04-23.md`) was a per-fold target-encoding leak hidden inside what looked like walk-forward CV. The fold structure was correct; the encoder fits weren't. Both pieces have to be honest.

---

## Current state (v3.1.0)

### Fold construction — `pipelines/04_build_folds.py`

5 temporal folds via `temporal_fold_{i}` columns in `data/train_test_splits.parquet`. Each fold is an (train, val, test) tuple where:

- **train** = stops with `stop_date < fold_train_end_date`
- **val** = stops with `fold_train_end_date <= stop_date < fold_val_end_date` (used for early-stop)
- **test** = stops with `fold_val_end_date <= stop_date < fold_test_end_date`
- **embargo** = 90 days between train and val (Track A from 2026-04-24 to prevent rolling-feature leakage; specifically `stops_last_90d` could carry train info into the val period)

| Fold | Train end | Val end | Test end | Notes |
|---|---|---|---|---|
| 0 | early — first quarter of data | … | … | Hardest fold; sparsest features, most regime shift |
| 1 | second quarter | … | … | First "real" fold with stable feature coverage |
| 2 | third quarter | … | 1 row | **Degenerate — only 1 test row. Skipped in train_fold (`train_multi.py`).** |
| 3 | fourth quarter | … | … | Solid fold; good coverage |
| 4 | most-recent | … | … | Best fold (most data + closest to production) |

### Per-fold encoder fits (Track A)

`FoldSafeTargetEncoder` (`data.py:36`) fits ON THE TRAIN SLICE ONLY for each fold. Application to val + test uses the train-fit smoothed lookup. Smoothing constant 20.0 prevents per-category mean overfitting.

**Critical**: validation set is the early-stop SIGNAL — fitting an encoder on val is the same as training on val. The Track A fix (2026-04-24) made this train-only across 8 files.

### Specialist checkpointing

Per fold, training writes:

- 1 composite-best checkpoint (`unified_phase2_fold{i}.pt`) — best epoch by composite metric
- 7 per-head specialist checkpoints (`unified_phase2_fold{i}_best_{metric}.pt`) — best epoch for each head individually

**Empirical finding** (Apr 23-24): per-head specialists beat composite-best at production inference. Speed head keeps improving 20+ epochs after the composite plateaus. Search/accident/injury peak in the first 0-2 epochs and then drift down. Specialist ensembling captures the per-head "best moment."

### Inference ensemble

`predict_multi_ensemble.py` loads 5 × 7 = 35 specialist checkpoints + 5 composite, scores all stops + bins through each, averages outputs across folds (not across head-types — each head's per-fold specialist averages with itself).

---

## Known gaps / pain points

- **Fold 2 degeneracy is silent**. `train_multi.py` skips it via length check. No alert if other folds become degenerate from data shifts.
- **No per-fold size table is documented anywhere**. Researchers can't easily see "fold 0 has X train stops, Y val, Z test" without reading the parquet.
- **No 6th held-out fold**. The 5 folds cover all data; production scoring goes against the same data the test sets are drawn from. We have no "future I never touched" benchmark.
- **Bin splits use random 80/20** (`07a_train_occurrence.py:110-113`), not temporal. XGB baselines on bin-level metrics are mildly optimistic.
- **No fold-stability metric**. We track per-fold composite individually but no automated check that "the model is stable across folds" — fold 0 at -0.21 vs fold 4 at +0.55 is a HUGE spread.
- **Embargo is 90 days, but rolling features go back 90 days** (`stops_last_90d`). Embargo should match feature lookback or exceed it. 90 = 90 means train data within 1-day feature lookback of the embargo-start can leak into val. Audit needed.

---

## Open questions

- Should embargo be 91 days (one day longer than longest rolling feature) to fully decouple?
- Should fold 2 be redesigned (extend test window forward) or accepted as a known degenerate?
- Should we report per-fold metrics in the user-facing UI ("model has +0.55 composite on data-similar-to-now, -0.21 on early data") rather than aggregate?
- Should specialist checkpoints be replaced by a learned-weighted average (small head over [composite, per-head specialists])?

---

## Next-gen direction

For v3.2.0 / Phase 3:

1. **Add fold 5: held-out future** — last 90 days of data never used in any fold's train/val/test. Provides a true "production-realistic" benchmark.
2. **Document per-fold sizes** (auto-generate table from `train_test_splits.parquet` into a section in this file or in [`09_evaluation_metrics.md`](09_evaluation_metrics.md)).
3. **Walk-forward bin splits** — rebuild XGB baselines with temporal bin splits to match the FT-T. Removes a known unfairness in the composite comparison.
4. **Stability metric**: alert if max(fold) - min(fold) composite > 0.5. Currently it IS that big (-0.21 to +0.55). Triggers per-fold inspection.
5. **Specialist ensembling justified** — write a small validation that per-head specialists actually beat composite-only on a held-out subset, before relying on it.
6. **Drop fold 2 explicitly** in code with a comment, or fix the temporal bounds. Silent skipping is fragile.

---

## References

- `docs/audits/walk-forward-cv-audit-2026-04-24.md` — the audit + Track A design
- `docs/audits/speed-auc-leak-investigation-2026-04-23.md` — the leak that motivated train-only encoder fits
- [`03_label_design.md`](03_label_design.md) — what the 7 heads target
- [`07_training_loop.md`](07_training_loop.md) — how the alternating multi-task loop interacts with CV

---

## TODO

- [ ] Add fold 5 (held-out future)
- [ ] Auto-generate per-fold size table from `train_test_splits.parquet`
- [ ] Rebuild XGB bin baselines with temporal splits
- [ ] Add fold-stability alert (max - min > 0.5 composite)
- [ ] Validate specialist-ensemble actually beats composite-only on held-out (defensive — confirm assumption)
- [ ] Audit embargo = 91 vs rolling-feature lookback (decide off-by-one)
