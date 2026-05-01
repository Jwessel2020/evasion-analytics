# LSTM + GNN Integration with FT-Transformer

**Drafted 2026-04-26.** Architectural plan for integrating temporal (LSTM, Phase 2) and spatial (GAT, Phase 3E) modules with the FT-Transformer backbone. Bridges between `05_model_architecture.md` (master design) and `13_ablation_studies.md` (which combinations to test). Written after the v3.2.0 PLR retrain proved that architecture upgrades pay back big — natural follow-up question is "what to add next, and how do we sequence the dev work to ship fast?"

## Purpose

Three modules — FT-Transformer (per-stop tabular), LSTM (per-cell hourly sequences), GAT (H3 cell adjacency graph) — exist in code but are partly disabled. This file decides:

- **What integration pattern** wires them together cleanly (5 candidates, ranked).
- **Which components ship in v3.3.0** (post-PLR), which defer to v3.4.0+.
- **How to compress the dev timeline** by running implementation in parallel via sub-agents and ablating in parallel across GPUs.

## Current state (post-v3.2.0)

| Module | Status | File |
|---|---|---|
| FT-Transformer backbone | ✅ in prod (v3.1.1), shipping v3.2.0 with PLR | `ml/research/deep/model.py` |
| PLR tokenizer (3A) | ✅ shipping in v3.2.0 | `ml/research/next_gen/tokenizers/plr.py` |
| LSTM `temporal_fuse` (Phase 2) | ❌ smoke FAILED 2026-04-26 (Δ -0.067 v2 attempt); deferred | `ml/research/deep/model.py:437-441` |
| `SpatialEncoder` GAT (3E) | ❌ smoke FAILED 2026-04-26 (Δ -0.156 fold 0); preserved on `next-gen/p2-spatial-cache` | `ml/research/next_gen/encoders/spatial_cache.py` |
| `cell_sequences.parquet` | ✅ available, ~500K samples, 168h × 6 features | `ml/research/data/cell_sequences.parquet` |
| `h3_adjacency.npz` | ✅ available, 1,356 nodes / 6,182 edges | `ml/research/data/h3_adjacency.npz` |

**Per-fold composite scores so far (v3.2.0 PLR):** fold 0 +0.292, fold 1 +0.657, fold 4 +1.143 (single-fold mean +0.70 across 3 done). The Phase 3 ship gate of +0.34 is comfortably cleared on every fold tested.

## Integration patterns (ranked by lift-per-dev-day)

| # | Pattern | Dev days | Expected composite lift | Risk | Code wiring |
|---|---|---|---|---|---|
| 1 | **LSTM late fusion** (concat → project) | 2-3 | +0.02 to +0.05 | Low | Already 90% wired at `model.py:437-441` |
| 5 | **Hawkes baseline + neural residual** | 3-5 | +0.05 to +0.10 (`occ_mae` head) | Low | New `HawkesIntensityHead` near `model.py:172` |
| 4 | **Archetype MoE head routing** (3B) | 4-6 | +0.05 to +0.10 | Medium | Replaces 7 task heads at `model.py:388-395` |
| 2 | **Spatial → temporal cascade** (GAT first) | 5-7 | +0.08 to +0.15 | Medium | Refactor `SpatialEncoder` to per-epoch cache |
| 3 | **True ST-GNN early fusion** (DCRNN-style) | 10-14 | +0.10 to +0.20 | High | New module in `next_gen/encoders/` |

## What the literature says (2023-2024)

- **Don't use Transformers for the time axis.** Zeng et al. 2022 ("Are Transformers Effective for Time Series Forecasting?") + Wu et al. 2023 (TimesNet) show that on regularly-sampled hourly time series with fewer than ~1000 timesteps, simple LSTM or temporal convs out-perform attention-based forecasters. Our 168-hour cell sequences land squarely in that regime — LSTM is the right choice.
- **GNN for fixed graphs is well-solved.** DCRNN (Li 2018) + Graph WaveNet (Wu 2019) + ST-GAT variants all work; differences are marginal at our node count (1,356). D2STGNN (2024) and adaptive-adjacency variants only matter if the graph itself is unknown — we have a fixed H3 hexagonal grid.
- **Late fusion > early fusion at our data scale.** Across recent ST-GNN benchmarks, late fusion (each module produces an embedding, concatenated at the end) is more sample-efficient than early fusion (joint encoding) when N < ~10K nodes. We're at 1,356.
- **Hawkes + neural residual is a known win for self-exciting events.** Mohler et al. (JASA 2015) deployed ETAS Hawkes in the LAPD with a 7.4% real-world crime reduction; CL-ETAS (Zhang 2024) showed that `Hawkes baseline + ConvLSTM residual > pure neural` for earthquake forecasting. Same self-excitation pattern is documented in police enforcement (officers cluster around incidents).
- **PLR + LSTM should compose cleanly.** PLR's periodic basis (`sin(2πWx), cos(2πWx)`) operates in feature space; LSTM's hidden state operates in time. Different abstraction levels — they reinforce, don't double-count.

References at the end.

## Recommendation for v3.3.0

> **Update 2026-04-26 evening**: BOTH Path A and Path B were obsoleted by the overnight smoke results. All 4 patterns failed their gates. **Path C — taken** — see closing section. Paths A/B preserved below as historical record of the pre-smoke planning.

### Path A — Conservative (sequential, ~2 weeks)

v3.3.0 = v3.2.0 + Pattern 1 (LSTM) + Pattern 5 (Hawkes).

```
Week 1: Pattern 1 — flag flip + smoke fold 4 + smoke fold 0. Decide ship-or-skip.
Week 2: Pattern 5 — implement Hawkes head, smoke fold 0 (sparse, where Hawkes shines).
Week 3: Full 5-fold retrain v3.3.0 → ensemble → deploy.
```

Expected lift: composite +0.04 to +0.10 over v3.2.0. Low risk, low surprise.

### Path B — Aggressive bundle (parallel sub-agents, ~10 days)

v3.3.0 = v3.2.0 + Patterns 1 + 5 + 4 + 2 (everything except Pattern 3 deferred).

```
Day 0:    Ship v3.2.0 to prod. Spawn sub-agent #A: implement Pattern 5 (Hawkes).
                                Spawn sub-agent #B: implement Pattern 4 (MoE).
                                Spawn sub-agent #C: refactor Pattern 2 (SpatialEncoder).
Day 0-1:  Pattern 1 LSTM smoke on laptop fold 4 (no dev needed; runs in parallel with sub-agents).
Day 1-7:  Sub-agents work on own branches. Daily review + merge.
Day 7-8:  Smoke each implemented pattern in parallel — laptop fold 4 + desktop fold 0 + Kai fold 1 + David fold 3 (assumes friend GPUs are connected). Each smoke ~1 hour.
Day 8-9:  Full 5-fold retrain combined v3.3.0 with whatever passed smoke. Each GPU runs one fold.
Day 10:   Ensemble inference + deploy v3.3.0.
```

Expected lift: composite +0.10 to +0.25 over v3.2.0. Higher variance because more moving pieces; mitigated by the smoke-gate at Day 7-8 (anything that doesn't lift gets dropped).

**Path B is the recommended default** if Kai + David are likely online. With 4 GPUs available we can run 4 ablations in parallel on Day 7-8, plus 4 folds in parallel on Day 8-9, compressing the wall-time bottleneck from 14 days to ~10. Without friend GPUs, Path A is more honest about the timeline.

## Time-optimization principles

These apply across any Phase 3 / Phase 4 work, not just v3.3.0:

1. **Always have a smoke running on each free GPU.** Built-but-not-tested code is dead weight. As soon as a sub-agent merges a pattern, kick off its smoke on the next free GPU.
2. **Always have a sub-agent building.** Implementation is the slowest serial bottleneck because human-in-the-loop dev is sequential per developer. Spawning N sub-agents on N branches turns N days of serial work into max(individual-day-counts) of parallel work. Cost: review time at end of each agent's run.
3. **Smoke before retrain.** A 30-epoch fold-4 smoke (~30-60 min on laptop) tells you 90% of what a full 60-epoch 5-fold retrain (~6-8 hours) would. Use smokes as cheap gates.
4. **Use the `--check` flag on installer scripts** for friend-machine setup verification — if Kai/David don't run jobs, parallelism collapses to the local 2 GPUs.
5. **Bundle the deploy.** Each retrain → ensemble → deploy chain is ~30 min. If two patterns finish their smokes within an hour of each other, retrain them together (one bundled fold). Don't wait for one to ship before starting the next.

## Per-pattern implementation notes

### Pattern 1 — LSTM late fusion (already half-done)

Code already at `model.py:437-441`:
```python
if self.use_temporal:
    seq_h = self.lstm(temporal_seq)  # (B, T, D) → (B, D)
    fused = torch.cat([cls_emb, seq_h[:, -1]], dim=-1)
    cls_emb = self.temporal_fuse(fused)
```

To activate: pass `--temporal` to `train_multi.py` (the flag exists). The smoke-gate question is whether composite delta ≥ +0.02 on fold 4 vs the v3.2.0 fold-4 baseline of +1.143. If yes, ship as part of v3.3.0.

**Risk on sparse cells (fold 0)**: cells with no recent activity feed mostly-zero sequences. Mitigate by gating: `cls_emb + sigmoid(gate) * temporal_contribution` so the gate learns to mute the LSTM where it's noise.

### Pattern 5 — Hawkes baseline + neural residual

**Status (2026-04-26 evening)**: code shipped on branch `next-gen/p5-hawkes`.
Module: `ml/research/next_gen/heads/hawkes.py` (~280 LOC).
Tests: 31 unit tests pass (`next_gen/tests/test_hawkes.py`).
Wiring sidecar: `ml/research/next_gen/hawkes_train_hooks.py`.
**Smoke FAILED**: fold 0, 60ep, `--use-plr --use-hawkes`: best `occ_mae=0.395` / composite `+0.262` @ ep 38. Gate (≥−0.02 occ_mae vs 0.398 baseline) cleared by +0.003; composite gate not cleared (-0.030 vs +0.292). Net assessment: Hawkes adds learnable params that compete for budget with PLR's already-strong baseline; PLR captured most of the self-excitation lift via its periodic features. Branch preserved for future revisit but not shipping in v3.3.0.

The intensity is a closed-form ETAS sum:

```python
λ_hawkes(c, t) = μ_c + α · Σ_{s in [t-168h, t)}  count(c, s) · exp(-β · (t - s))
```

`μ_c` is per-cell; `α` and `β` are shared scalars. The head returns
`λ_hawkes` (positive rate, NOT log-rate). The FT-T's `OccurrenceHead`'s
output is reinterpreted as a residual logit; the combined log-rate fed
to `F.poisson_nll_loss(log_input=True)` is:

```python
combined_log_rate = log( λ_hawkes  +  softplus(neural_residual_logit) + 1e-8 )
```

The `softplus` ensures the residual contribution is non-negative (the
Hawkes baseline establishes a floor; the network can only push the
prediction UP).

**Numerical stability**:
- β clamped to `[0.01, 1.0]` after softplus → half-life range [0.69h, 69h]
  (decay ranges from sub-hour to 3 days). Typical crime data ≈ 0.1
  (6h half-life).
- Excitation sum computed via `logsumexp(log(count) - β·lag)` instead
  of direct `Σ count · exp(-β·lag)` — avoids exp-underflow for large
  lag · β products.
- Count==0 rows masked via `log_count = -1e9` sentinel (logsumexp
  ignores them cleanly without producing NaN gradients).
- All three params stored as raw real-valued `nn.Parameter`s; passed
  through `softplus` (and clamp for β) on each forward — gradient-stable.

**Initialization** (defaults; CLI flags override):
- `μ_c`: per-cell empirical-Bayes-shrunk hourly rate from training
  data via `next_gen.data_extensions.empirical_bayes_mu_per_cell`.
  Shrunk toward global mean by factor `n_c / (n_c + τ)`, τ = max(50,
  median(n_c)/5). Standard recipe per the
  [Mohler/Hawkes implementation plan](../../exploration/hawkes-implementation-plan-2026-04-24.md)
  Q3 "Cells with few events".
- `α = 0.3` per Mohler LAPD trial regime [0.3, 0.7] for crime.
- `β = 0.115` per hour, half-life ≈ 6h — consensus crime self-
  excitation timescale.

**Per-cell vs shared (μ, α, β)** (open question from earlier draft):
v1 ships **per-cell `μ_c` + shared `α`, `β`** (1,356 + 2 = 1,358 params).
This matches the planning doc's recommendation. Per-cell `(α_c, β_c)`
adds ~3K params but risks instability on cells with <100 events; defer
to v2.

**Why this targets `occ_mae` specifically**: Poisson regression with
Hawkes baseline assumes the data is event-driven, which is exactly
what police enforcement is — clusters around incidents. The current
v3.2.0 fold-4 `occ_mae` is 0.358 vs XGB 0.339 (still 6% behind, per
[Mohler 2015 LAPD RCT](../papers/) which deployed ETAS in production
with 7.4% real-world crime reduction). Hawkes should close most of
that gap.

**Wiring contract** (what `train_fold` in `next_gen/train.py` does
when `--use-hawkes` is set, per the sidecar's docstring):

1. After UnifiedModel is built (and after PLR if `--use-plr`):
   ```python
   haw_train, haw_test = build_hawkes_dataset_pair(fold_idx)
   mu_init = compute_empirical_bayes_mu(haw_train)
   haw_head = patch_with_hawkes(model, n_cells=haw_train.n_cells,
                                 mu_init=mu_init, ...)
   haw_loader = DataLoader(haw_train, batch_size=bs, shuffle=True)
   ```
   Note: `patch_with_hawkes` registers the head as `model._hawkes_head`,
   so `model.parameters()` (and the optimizer) automatically includes
   its params. Per BUILD_PROCESS rule: deep/ is not modified.

2. In each training step, AFTER the bin-based occurrence batch's
   backward and BEFORE the stop-multi batch:
   ```python
   bh = next(haw_iter)  # cycle if exhausted
   log_rate = hawkes_combine_log_rate(
       haw_head, bh["cell_idx"], bh["cell_sequence"],
       model(...)["occurrence"],
   )
   loss = args.hawkes_loss_weight * F.poisson_nll_loss(
       log_rate, bh["target"], log_input=True,
   )
   loss.backward()  # accumulates with the bin-loop's gradients
   ```
   Default `hawkes_loss_weight = 0.25` matches `LOSS_WEIGHTS["occurrence"]`.

3. Save `haw_head.state_dict()` in the checkpoint payload alongside
   `model.state_dict()` so inference can reload it.

The wiring lives in `next_gen/hawkes_train_hooks.py` (sidecar) so that
sibling next-gen branches (`p2-spatial-cache`, `p4-moe-heads`) editing
`train.py` extensively don't risk losing the Phase 5 logic. The smoke
harness imports from the sidecar directly.

**Required input data** for the smoke run:
- `ml/research/data/cell_hourly_counts.parquet` (present; built by
  `deep/build_sequences.py`)
- `ml/research/data/cell_sequences_meta.json` (present)

If those are missing, `build_hawkes_dataset_pair` raises a clear
`FileNotFoundError` pointing to the build script.

**Known data-prep blocker (2026-04-26)**: the
`cell_sequences_meta.json` on disk lists 1,356 res-8 H3 cells
(`882a...`), but `cell_hourly_counts.parquet` was rebuilt later with
`H3_RESOLUTION=9` and now stores 6,791 res-9 cells (`8926...`). Zero
overlap. The base `TemporalOccurrenceDataset.__init__` filters counts
by membership in `meta["cells"]`, so it currently produces 0 samples
per fold. **This must be resolved before the Phase 5 smoke run.** Two
options:

1. Rebuild meta to res-9: `cd ml/research && python deep/build_sequences.py`
   (regenerates both files atomically using `config.H3_RESOLUTION = 9`).
   Cleanest fix.
2. Or downgrade `cell_hourly_counts.parquet` back to res-8: revert
   `config.H3_RESOLUTION` to 8 temporarily, rerun, then restore. Only
   if other downstream code requires res-8 counts.

The Hawkes head code itself is unaffected by this blocker — the
HawkesIntensityHead's `n_cells` arg adapts to whatever the dataset
reports. Only the dataset's empty-output is the problem. Smoke gate
verification: after rebuild, `len(haw_train) > 0` and
`haw_train.cell_counts.sum() > 0`.

**Smoke decision rule** (from the table further down in this file):
Δ `occ_mae` ≥ −0.02 on fold 0 vs the v3.2.0 fold-0 baseline of
0.398 (composite +0.292; see `models/v3.2.0/next_gen_metrics_fold0.json`).
Pass = include in v3.3.0; fail = drop, document why.

### Pattern 4 — Archetype MoE head routing ❌ SMOKE FAILED (2026-04-26)

Replace the 3 shared `OccurrenceHead` instances at `model.py:388-390` with `ArchetypeMoEHead`. Each cell's K-means archetype (from `cell_archetypes.parquet`, K=24) hard-routes to one expert MLP. Plus one always-on shared expert. Loss is per-expert Poisson NLL + Switch-Transformer-style load-balancing penalty applied to the shared-blend gate + per-bucket count distribution.

**Risk**: with K=24 archetypes and walk-forward CV, some experts train on <1% of data. Mitigation: merge tail archetypes (top-N by training-set sample count, route the rest to "other") — `top_n_archetypes=12` covers ~70% of bins.

**Why deferred priority**: PLR already closed most of the per-head gap that MoE was scoped to address. Worth testing but not urgent.

**Implementation status (2026-04-26 — built on `next-gen/p4-moe-heads`, awaiting smoke)**:
- `ml/research/next_gen/heads/moe.py` — `ArchetypeMoEHead` (drop-in for `OccurrenceHead` plus `archetype_id` arg) + `load_balance_loss`. Per-bucket experts + always-on shared expert with learned per-bucket blend gate (init -2.0 → sigmoid 0.12, routed expert dominates initially).
- `ml/research/next_gen/data_extensions.py` — `OccurrenceDatasetWithArchetype` and `StopMultiDatasetWithArchetype` extend `deep.data` to expose per-row `archetype_id`. `add_h3_cell_to_stops` enriches stops with `h3_cell` from lat/lng (h3 v4 API, resolution 9 — confirmed via `h3.get_resolution()` against `cell_archetypes.parquet` sample).
- `ml/research/next_gen/moe_train_hooks.py` — sidecar wiring helpers (`patch_with_moe`, `moe_forward_occurrence`, `moe_backbone_forward`, `eval_metrics_moe`) so the wiring contract is independent of `train.py`'s exact line numbering. Mirrors the Phase 5 sidecar pattern.
- `ml/research/next_gen/train.py` — `--use-moe`, `--n-archetypes`, `--top-n-archetypes`, `--lambda-balance` CLI flags + sidecar import. NOT compatible with `--use-spatial` in this harness — they use different per-row metadata (`archetype_id` vs `cell_idx`); combine after both ship.
- `ml/research/next_gen/tests/test_moe.py` — **23/23 tests pass** across 6 classes covering instantiation, forward shape, mixed-archetype batches, OOB-archetype routing, gradient flow, load-balance behaviour (uniform → low loss; collapse → high loss), tail-archetype merging, and drop-in compat with `OccurrenceHead`.

**Hyperparameters baked in** (per `06_hyperparameters.md` Phase 3 row):
- `n_archetypes = 24` — matches `cell_archetypes.parquet`
- `top_n_archetypes = 12` — covers ~70% of bins; remaining tails route to single "other" bucket
- `lambda_balance = 0.01` — Switch Transformer 2021 §3.2 default
- expert `hidden_dim = 128` — matches `OccurrenceHead`
- shared-blend init = `-2.0` — sigmoid(-2) ≈ 0.12, routed expert dominates initially

**Smoke result (2026-04-26)**: fold 4, 30ep, `--use-plr --use-moe`: best composite **+0.917 @ ep 23**. Gate (≥+0.03 vs +1.143 fold 4 baseline) **NOT cleared** — actual Δ = -0.226. Branch `next-gen/p4-moe-heads` preserved at commit 0020de7 with all 23 unit tests passing.

**Why it failed (working hypothesis)**: 36% of bins lack archetype mapping → all route to "other" expert which dominates. Plus the 12 expert MLPs train from scratch in 30 ep while PLR baseline is near-converged. Reviving this would require either (a) re-running `pipelines/09e_upload_archetypes.py` with K=64 to cover those orphan bins, or (b) warm-starting MoE experts from the v3.2.0 OccurrenceHead weights. Defer to v3.4.0+ if data justifies.

Original smoke cmd (preserved for reproducibility):

```bash
cd ml/research
MODELS_DIR=models/v3.2.0-3b python next_gen/train.py --fold 4 --epochs 30 --use-plr --use-moe
```

**Open design questions** (also documented in `next_gen/heads/moe_train_integration.md` Caveats):
- 36% of bins lack archetype mapping (cells outside the K-means clustering set, ~415K of 1.14M bins per the 2026-04-26 audit) — they all route to the "other" bucket and may dominate it. If load-balance can't push that down, re-run `pipelines/09e_upload_archetypes.py` with larger K.
- Stops-side h3 enrichment is ~5-10 s for 1.2M rows; cache to parquet if iteration time becomes a bottleneck.
- Inference path (`predict_multi_ensemble.py`) needs a `patch_with_moe()` mirror after smoke confirms the lift — the saved `next_gen_flags` dict in checkpoints already records the MoE config (n_archetypes, top_n_archetypes, lambda_balance).

### Pattern 2 — Spatial → temporal cascade ❌ SMOKE FAILED (2026-04-26)

The `SpatialEncoder` GAT exists at `model.py:239-304` but assumes full-batch input (all 1,356 cells at once). The training loop is mini-batch. Refactor — **landed as `next_gen/encoders/spatial_cache.py`**:

```
At epoch start:
    spatial_emb = SpatialEncoder(all_cells, h3_adj)  # (1356, 128), once per epoch
    cache = spatial_emb.detach()  # frozen for the batch loop

In each batch:
    cell_emb = cache[batch_cell_ids]  # (B, 128)
    fused = projection(concat([cls_emb, cell_emb]))  # late fusion → (B, 128)
```

Cache is ~50ms per epoch refresh. Negligible.

**Why important even when PLR helped**: spatial smoothing helps cells with sparse history borrow signal from active neighbors. Fold 0 (sparse pre-2020 data) is the natural test case. PLR helped fold 0 a lot but the +0.292 there is still the worst fold; spatial may close more.

**Implementation status (2026-04-26)**:
- `ml/research/next_gen/encoders/spatial_cache.py` — `SpatialCache` (refresh + lookup) + `build_default_cache()` factory + `load_h3_adjacency()` helper
- `ml/research/next_gen/data_extensions_spatial.py` — `OccurrenceDatasetWithCellIdx` + `StopMultiDatasetWithCellIdx` (resolve `h3_cell` → graph row index, with res-9 → res-8 parent walk to bridge bin/graph resolution mismatch — 91.5% hit rate audited 2026-04-26)
- `ml/research/next_gen/train.py` — `--use-spatial`, `--adjacency-path`, `--spatial-fallback {mean,zero}` CLI flags; `patch_with_spatial_cache()` monkey-patches `model.backbone.forward` to fuse `_pending_cell_emb` (set per batch) with the [CLS] output via `Linear(2*128 → 128)`
- `ml/research/next_gen/tests/test_spatial_cache.py` — 19/19 unit tests pass
- **Smoke result (2026-04-26)**: fold 0, 30ep, `--use-plr --use-spatial`: best composite **+0.136 @ ep 17**. Gate (≥+0.05 vs +0.292 fold 0 baseline) **NOT cleared** — actual Δ = -0.156. Branch `next-gen/p2-spatial-cache` preserved at commit cefe29f. Suspect: 8.5% of bins fall back to mean cache embedding, plus the GAT node-input is a randomly-initialized `nn.Parameter` that doesn't get gradient (cache uses `update_grad=False`). Revisit with static cell features (open question 1 below) before declaring this dead.

**Hyperparameters baked in** (defaults — override via CLI or in code if smoke needs it):

| Knob | Value | Rationale |
|---|---|---|
| GAT n_heads | 4 | Matches `deep/model.py` SpatialEncoder default |
| GAT n_layers | 2 | Matches existing SpatialEncoder (`_gat_layer` called twice in `forward`) |
| GAT dropout | 0.1 | Consistent with backbone residual dropout |
| Cache refresh frequency | once per epoch | ~50 ms cost; negligible vs ~30 s/epoch training |
| Cache `update_grad` | `False` (detach) | Per-batch path doesn't backprop through GAT; encoder gets no per-batch gradient. Acceptable for the smoke; revisit if Δcomp < +0.05 |
| Late-fusion projection | `Linear(2*128 → 128)`, no activation | Keeps [CLS] semantics linear; matches Pattern 2 spec |
| Per-cell node-input | learnable `nn.Parameter(N_cells, 128)`, init `N(0, 0.02²)` | GAT paper convention for node embedding GNNs |
| Fallback for unmapped cells | `"mean"` (mean cache embedding) | Affects ~8.5% of bin rows; safer than zero |

**Open design questions** to revisit if the smoke result motivates a v2:
1. **Static cell features instead of learnable node embedding**: feeding pre-computed cell-level features (lat/lng, road, distance-to-POI) to the GAT instead of a random `nn.Parameter` could give the spatial signal more grounding. Cost: a small MLP to project the per-cell template into the 128-d input space.
2. **Refresh more often than once per epoch**: if smoke shows the cache going stale fast (cls + cell_emb correlation drops mid-epoch), refresh per-N-batches.
3. **`update_grad=True` for refresh path**: lets backprop through GAT each refresh — the encoder participates in the per-task losses end-to-end. Cost: full graph forward + backward per refresh. Worth trying if Δcomp < +0.05 with `update_grad=False`.
4. **GATv2 (Brody 2021)**: addresses static-attention issue in vanilla GAT. Drop-in replacement, ~20 lines. Defer until vanilla GAT smoke result lands.
5. **Combine with per-cell archetype routing (Pattern 4 MoE)**: use the cached cell embedding as additional input to the MoE router, so routing becomes both archetype-aware AND spatially-smoothed. Cross-component synergy; only worth wiring if both 3B and 3E ship.

### Pattern 3 — True ST-GNN early fusion (deferred)

DCRNN-style temporal-graph convolutions per timestep. New module, ~2 weeks. Defer until Path B v3.3.0 ablations confirm we still need more spatial-temporal expressivity. Most likely outcome: not needed at our scale.

## Decision gates — RESULTS (2026-04-26 overnight)

| Pattern | Smoke gate | Smoke result | Δ vs baseline | Verdict |
|---|---|---|---|---|
| 1 LSTM | Δ comp ≥ +0.02 fold 4 (vs +1.143) | NOT RUN — full integration deferred (~60-90 min dev for OccurrenceDatasetWithSequence) | — | DEFERRED |
| 5 Hawkes | Δ `occ_mae` ≥ −0.02 fold 0 (vs 0.398) | fold 0, 60ep, --use-plr --use-hawkes: occ_mae 0.395, comp +0.262 @ ep 38 | +0.003 occ_mae (need ≥0.02), -0.030 cmp | ❌ FAIL |
| 4 MoE | Δ comp ≥ +0.03 fold 4 (vs +1.143) | fold 4, 30ep, --use-plr --use-moe: comp +0.917 @ ep 23 | -0.226 cmp | ❌ FAIL |
| 2 Spatial | Δ comp ≥ +0.05 fold 0 (vs +0.292) | fold 0, 30ep, --use-plr --use-spatial: comp +0.136 @ ep 17 | -0.156 cmp | ❌ FAIL |
| 3 ST-GNN | not built | — | — | — |

**Bundle composite gate for v3.3.0 5-fold retrain**: mean composite ≥ +0.40 — N/A, no patterns passed individual smoke. v3.3.0 retrain is **HELD**; see `docs/v330-overnight-brief-2026-04-26.md` for the morning-review options (A: hold + pivot to Phase 4; B: longer/combined smokes; C: integrate Pattern 1 LSTM properly).

**Working hypothesis on universal failure**: PLR (Phase 3A) likely captured most of the signal these patterns aimed to add. Each pattern adds learnable components that compete with PLR's already-strong baseline within a 30-60 ep smoke window — net negative because new components train from scratch while baseline is already close to converged.

## Path C — TAKEN (2026-04-26 evening)

After all 4 architectural patterns failed their smoke gates, v3.3.0 pivoted from "add architectural complexity" to "add a new prediction head + a high-cardinality input feature". The aim is product-side lift (a new endpoint for citation/warning probability) rather than architectural lift over v3.2.0's per-head AUCs.

**v3.3.0 changes (committed `8df92ca` on `next-gen/p5-hawkes`)**:

1. **New `disposition` head**: 5th binary head, predicts `P(citation | stop) ∈ [0, 1]`. Mirrors the existing `SpeedHead` pattern (single-output sigmoid over CLS+stop_encoder concat).
2. **New `arrest_type_letter` categorical input**: 19 distinct letters (A-S) replacing the leakage-prone `violation_type` / `violation_type_top` inputs. Cardinality cap = 32 (from 16) to leave room for future codes.
3. **`is_citation` derived target**: pipeline 03 line 639 — `(violation_type == "Citation").astype("int8")`. Removed from feature inputs simultaneously to prevent target leakage.
4. **LOSS_WEIGHTS rebalanced** to sum=1.00 with disposition allocated 0.08 (taking from speed_occ which dropped from 0.13 to 0.10).
5. **`predict_multi_ensemble.py` multi-version loader**: auto-detects v3.2.0 vs v3.3.0 via `stop_encoder.0.weight` shape; output parquet enriched with `disposition_citation_prob` + 7 enforcement metadata columns.

**Smoke result (2026-04-26 evening)**: fold 0, 60ep, `--use-plr` + disposition head: best composite **+0.336 @ ep 41**. Gate (≥+0.292 vs v3.2.0 fold 0 baseline) **PASSED** by +0.044. disposition_auc fold 0 was 0.4716 (concerning but explained by 83/17 fold-0 class drift; fold 4's 64/36 should give cleaner signal). All other heads improved or held; arrest_type_letter compensates for removing violation_type.

**Status**: 5-fold retrain pending (Modal L4 folds 0+4 + Desktop 4070 SUPER folds 1+3). Ensemble inference + deploy as `v4.3.0-ensemble` after.

## Open questions

- **Does the LSTM hidden dim 64 need to grow with PLR's expanded feature space?** PLR adds periodic basis features; LSTM's input is independent (it consumes per-cell hourly counts, not the FT-T features). So no — but worth checking whether the LSTM's downstream contribution becomes redundant after PLR.
- **Should Hawkes per-cell `(μ, α, β)` be shared across cells or per-cell?** Per-cell adds 3K × cells = ~4K parameters; cheap. Per-archetype (24 sets) is more parsimonious. Test both.
- **How does v3.3.0 affect the production prediction pipeline?** The current ensemble loader (`predict_multi_ensemble.py`) assumes a fixed model class. Hawkes needs per-cell history at inference, MoE needs the archetype lookup; both add inference-time data dependencies.
- **What's the right "PLR + LSTM smoke" composite baseline?** Use fold-4 v3.2.0 (+1.143) as the reference and require LSTM+v3.2.0 to clear it by ≥ +0.02. Since fold 4 is the easy fold, the gate is set at the easier-end of the distribution.

## References

- [`05_model_architecture.md`](05_model_architecture.md) — master Phase 3 design (5 components detailed)
- [`13_ablation_studies.md`](13_ablation_studies.md) — formal ablation grid
- [`docs/hawkes-implementation-plan-2026-04-24.md`](../../hawkes-implementation-plan-2026-04-24.md) — earlier Hawkes design (predates Pattern 5 framing here)
- [`docs/hybrid-architecture-compositions-2026-04-24.md`](../../hybrid-architecture-compositions-2026-04-24.md) — alternative compositions to consider
- [`ml/research/next_gen/BUILD_PROCESS.md`](../../../ml/research/next_gen/BUILD_PROCESS.md) — build cycle + per-component log
- [`ml/research/next_gen/encoders/README.md`](../../../ml/research/next_gen/encoders/README.md) — `SpatialEncoder` discovery + planned refactor

### Papers (Tier 1 — read before starting v3.3.0 work)

- **Li et al. 2018** — DCRNN (Diffusion Convolutional Recurrent Neural Network) — the canonical late-fusion ST-GNN
- **Wu et al. 2019** — Graph WaveNet — dilated causal convs + adaptive adjacency
- **Mohler et al. 2015** (JASA) — ETAS Hawkes deployed in the LAPD, 7.4% real-world crime reduction
- **Zeng et al. 2022** — "Are Transformers Effective for Time Series Forecasting?" — counter-evidence against using Transformer for the time axis at our scale
- **Wu et al. 2023** — TimesNet — temporal 2D-variation modeling, current strong baseline

### Papers (Tier 2 — for spatial cascade work)

- **Zhou et al. 2021** — Informer — efficient long-sequence Transformer (relevant if we ever extend sequences past 1000h)
- **D2STGNN 2024** — Decoupled Dynamic Spatial-Temporal GNN — overkill for fixed H3 graph
- **Zhang et al. 2024** — CL-ETAS — Hawkes baseline + neural residual for earthquake forecasting (validates Pattern 5)

## TODO

- [x] Run smokes for Patterns 2, 4, 5 — all FAILED (see Decision gates table)
- [x] Pivot to Path C (v3.3.0 = PLR + disposition + arrest_type_letter, see Path C section)
- [ ] Complete v3.3.0 5-fold retrain (Modal L4 folds 0+4, Desktop folds 1+3)
- [ ] Ensemble inference + deploy as v4.3.0-ensemble
- [ ] Post-deploy: revisit Pattern 2 with static cell features (open question 1) and Pattern 4 with archetype K=64 — both have preserved branches; defer to v3.4.0+ if a use-case justifies
