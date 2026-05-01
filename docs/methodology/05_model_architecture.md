# Model Architecture

**Drafted 2026-04-25.** FT-Transformer backbone + 7 task heads + optional Phase 2 LSTM + optional Phase 3 GAT. Where every architectural choice lives in code, what's actually active in v3.1.0, and the 5-component Phase 3 roadmap (PLR + MoE + decorrelation + head-attention + GAT).

---

## Purpose

Architecture is where v3.1.0 hits its current ceiling. Per-head AUCs trail XGB on most heads (composite +0.29 mean wins, but per-head we're still catching up on speed/accident/injury). The hypothesis: feature/label fixes (Phase 0) unlocked the composite gain; architecture upgrades (Phase 3) unlock per-head gains.

---

## Current state (v3.1.0)

### Backbone: FT-Transformer (Gorishniy 2021)

`UnifiedModel` in `ml/research/deep/model.py`:

```
inputs (cell_numeric, cell_categorical, time)
   ↓ NumericTokenizer    (model.py:28-41)    one Linear per feature → tokens
   ↓ CategoricalTokenizer                    embedding + Linear → tokens
   ↓ + TimeTokenizer
   ↓ prepend [CLS] token
   ↓ × 3 TransformerBlock                    pre-LN MHSA + ReGLU FFN + residual
   ↓ LayerNorm(tokens[:, 0])                 [CLS] readout
   ↓ Linear(d_token → 128)                   backbone output
```

**Key details**:
- `d_token = 64` actual (occasionally documented as 128 elsewhere — VERIFY)
- `n_layers = 3`, `n_heads = 4`, `d_ffn = int(64 × 4/3) ≈ 85`
- Dropout: 0.2 attention, 0.1 residual (residual dropout added 2026-04-24)
- Pre-norm (not post-norm). Kaiming uniform init, zero bias.
- ReGLU FFN (gated, not standard ReLU): `[gate, value] = chunk(W·x); out = ReLU(gate) ⊙ value; W'·out`. ~10% better than standard FFN per FT-T paper.

### Heads — 7 shallow MLPs

```python
class Head(nn.Module):
    def forward(self, backbone_128):
        x = self.fc1(backbone_128)  # 128 → 128
        x = F.relu(x)
        x = self.fc2(x)              # 128 → 1
        return x
```

3 Poisson heads (occurrence, speed_occurrence, trap) + 4 binary heads (speed, search, accident, injury). All 7 share the backbone but each has its own MLP. No cross-head attention or routing.

### Optional Phase 2 LSTM — built but disabled

`build_sequences.py` produces 168-hour (7-day) sliding windows per cell with 6 features (`stop_count, hour_sin/cos, dow_sin/cos, is_weekend`). LSTM in `model.py:214-234`: 64-d hidden, 2 layers. Fused with backbone via `Linear(128+64 → 128)`.

**Activation**: `--temporal` flag in `train_multi.py`. Default OFF. Phase 2 production runs (the 5-fold v3.1.0 ensemble) used it.

### Optional Phase 3 GAT — built but disabled

`build_graph.py` produces `h3_adjacency.npz` (1356 nodes, 6182 edges via H3 grid_ring). `SpatialEncoder` in `model.py:239-304`: 2-layer GAT, 4 heads, ELU activation, residual connections, edge-index pre-built.

**Activation**: `use_spatial=True` flag. Default OFF. Never activated in production.

---

## Where every choice lives — code map

| Component | Location | Lines |
|---|---|---|
| `NumericTokenizer` | `model.py` | 28-41 |
| `CategoricalTokenizer` | `model.py` | 44-78 |
| `TransformerBlock` (pre-LN, ReGLU) | `model.py` | ~85-145 |
| `[CLS] readout` | `model.py` | 166 |
| Phase 2 LSTM `temporal_fuse` | `model.py` | 214-234 |
| Phase 3 GAT `SpatialEncoder` | `model.py` | 239-304 |
| `UnifiedModel` orchestration | `model.py` | 309-461 |
| 7 task heads (`OccurrenceHead`, `SpeedHead`, etc.) | `model.py` | 388-461 |
| Backbone forward path | `model.py` | ~310-385 |

---

## Known gaps / pain points

- **Per-task heads are shallow MLPs (128 → 128 → 1)**. No head-specific attention — every head sees the same backbone summary. Speed head can't preferentially weight road-speed features over crash features.
- **SpatialEncoder built but never activated**. h3_adjacency.npz exists, code path exists, no production retrain has used it. ~10% epoch overhead is the only cost.
- **No CKA / decorrelation tracking**. Speed head and trap head outputs may re-correlate during training even with mutually-exclusive labels (Phase 0C). Not measured; can't even tell if it's happening.
- **No uncertainty quantification anywhere**. No Bayesian layers, no MC-dropout at inference, no conformal calibration. Single-point predictions only.
- **Per-fold ckpt size = 35 specialists + 1 composite**. 40 files per fold × 5 folds = 200 .pt files per model version. Storage bloat; deploy script complexity.
- **`d_token=64` may be too small** for 247 features. FT-T paper recommends d_token ∈ [32, 128] but with the cell+time+per-stop dimensions we have, 64 may saturate.

---

## Open questions

- Is `d_token = 64` chosen by ablation or convenience? Worth a one-fold sweep over {32, 64, 96, 128}.
- Does activating SpatialEncoder help fold 0 (the hard regime)? Hypothesis: YES because sparse cells borrow signal from H3 neighbors.
- Should the [CLS] readout include the cell embedding directly (concat or add), giving the backbone a non-attention path to spatial signal?
- Is ReGLU strictly better than SwiGLU here? FT-T paper uses ReGLU; Llama uses SwiGLU. One-fold ablation could decide.

---

## Build status — Phase 3 components (updated 2026-04-25 19:35)

| Component | Status | Files | Tests | Smoke (fold 4) |
|---|---|---|---|---|
| **3A PLR tokenizer** | ✅ **SHIP** | `next_gen/tokenizers/plr.py` | 9/9 pass | **+1.036** (Δ +0.310 vs +0.726 baseline) |
| **3B MoE Poisson heads** | ✅ CODE READY (smoke pending) | `next_gen/heads/moe.py` + `next_gen/data_extensions.py` + `next_gen/moe_train_hooks.py` (sidecar, 2026-04-26) | 23/23 pass | smoke gate: fold-4 Δcomp ≥ +0.03 vs v3.2.0 +1.143 |
| 3C head-attention masks | not started | — | — | — |
| **3D decorrelation loss** | ⚠️ **DON'T SHIP STANDALONE** | `next_gen/losses/decorrelation.py` | 15/15 pass | +0.618 (Δ -0.108) — decorr never fires |
| 3E spatial GAT | ✅ CODE READY (smoke pending) | `next_gen/encoders/spatial_gat.py` + `next_gen/encoders/spatial_cache.py` (Pattern 2 wrapper, 2026-04-26) | 19/19 pass (cache); 9/9 pass (GAT) | smoke gate: fold-0 Δcomp ≥ +0.05 vs v3.2.0 +0.292 |

### 3A smoke result (2026-04-25)

Ran `next_gen/train.py --fold 4 --epochs 30 --use-plr` against `models/v3.1.1-3a/`.

| Metric | v3.1.1 baseline (vanilla) | v3.1.1-3a (PLR) | Δ |
|---|---|---|---|
| **Composite (best)** | +0.726 @ ep 16 | **+1.036 @ ep 18** | **+0.310** |
| occ_mae | 0.413 | **0.363** (-0.050) | better |
| speed_occ_mae | 0.080 | **0.030** (-0.050) | better |
| trap_mae | 0.056 | **0.051** (-0.005) | better |
| speed_auc | 0.817 | **0.822** (+0.005) | better |
| search_auc | 0.908 | 0.927 (+0.019) | better |
| accident_auc | 0.874 | 0.898 (+0.024) | better |
| injury_auc | 0.863 | 0.865 (+0.002) | tied |

Convergence: PLR hit baseline composite (+0.726) at epoch 4 (vs vanilla's epoch 16). **~4× faster convergence**, plus a higher peak. Every metric strictly improved or tied.

**Decision (per BUILD_PROCESS Step 5)**: Δ +0.310 >> +0.05 → SHIP. Combined Phase 3 retrain gets PLR.

#### Per-head specialist epochs (final 30-epoch run)

The composite-best epoch is the SAME for the model overall, but each head has its own peak — captured by `next_gen/train.py`'s specialist checkpoints:

| Head | Best value | Best epoch | Notes |
|---|---|---|---|
| `occ_mae` | 0.3634 | **18** | aligns with composite-best |
| `speed_occ_mae` | 0.0293 | **25** | still improving in later epochs |
| `trap_mae` | 0.0496 | **21** | mid-late peak |
| `speed_auc` | 0.8234 | **23** | binary heads with high pos_count keep climbing |
| `search_auc` | 0.9404 | **3** | early peak — typical pattern (matches vanilla) |
| `accident_auc` | 0.9218 | **2** | early peak (matches vanilla) |
| `injury_auc` | 0.9104 | **2** | early peak (matches vanilla) |

Pattern matches the vanilla v3.1.1 finding that the rare-positive binary heads peak in the first 2-3 epochs then drift down — specialist ensembling captures each head's "best moment." Phase 3 retrain should keep using per-head specialist checkpoints at inference, not just composite-best.

Cmd reproduced: `MODELS_DIR=models/v3.1.1-3a python next_gen/train.py --fold 4 --epochs 30 --use-plr`. 30 epochs ran to completion (no early stop — 30-cap reached). Wall: ~50 min on laptop A4000. Final composite +0.940 at ep 29 (drifted down from peak; model overfits late, specialist ckpts already captured peaks).

---

### 3D smoke result (2026-04-25 ~20:50)

Ran `MODELS_DIR=models/v3.1.1-3d python next_gen/train.py --fold 4 --epochs 30 --decorr` (vanilla model + decorr loss only, NO PLR).

| Metric | v3.1.1 baseline | v3.1.1-3d (decorr only) | Δ |
|---|---|---|---|
| **Composite (best)** | +0.726 @ ep 16 | +0.618 @ ep 9 | **-0.108** |

**Decision (per BUILD_PROCESS Step 5)**: Δ -0.108 < -0.02 → **DON'T SHIP STANDALONE**.

**Mechanism finding** — why decorr doesn't help:

The decorr loss `λ × ReLU(|corr| - 0.80)²` only fires when `|corr(speed_occurrence, trap)| > 0.80`. We measured |corr| every epoch via the harness's `decorr_history`:

| Epoch | λ_decorr | avg \|corr\| | Active? |
|---|---|---|---|
| 0-19 | 0.000 | 0.000 (not measured — λ=0) | inactive (annealing window) |
| 20 | 0.050 | **0.273** | ON, but corr ≪ 0.80 threshold |
| 21 | 0.055 | 0.275 | ON, no-op |
| 22 | 0.060 | 0.279 | ON, no-op |
| 23 | 0.065 | 0.266 | ON, no-op |
| 24 | 0.070 | 0.279 | ON, no-op |

**The Phase 0C label partition (mobile vs stationary, 2026-04-24) already decorrelated speed_occurrence and trap at the DATA level — measured |corr| ~0.27, well below the 0.80 threshold the loss would penalize.** The loss has nothing to bite on; it's effectively a no-op. The slight composite drop (-0.108) is training noise from the harness re-init, not from the loss term.

**Implication for Phase 3**: 3D is NOT needed for the data we have. Phase 0C does the job at the label level.

**Keep the code anyway** — `next_gen/losses/decorrelation.py` stays in the repo because:
1. It's a defensive tool if a future feature change reintroduces correlation between heads
2. The CKA tracking + `anneal_lambda_decorr` utility are reusable for other heads
3. `cka_linear` is a useful diagnostic regardless of decorr loss

Cmd reproduced: `MODELS_DIR=models/v3.1.1-3d python next_gen/train.py --fold 4 --epochs 30 --decorr`. Wall: ~50 min on laptop A4000.

#### 3D per-head specialist epochs (interesting — MIXED, not uniformly worse)

| Head | Vanilla baseline | 3D-only | Δ | Best epoch (3D) |
|---|---|---|---|---|
| occ_mae | 0.413 | 0.439 | -0.026 (worse) | 20 |
| speed_occ_mae | 0.080 | **0.033** | +0.047 (much better) | 18 |
| trap_mae | 0.056 | 0.061 | -0.005 (slightly worse) | 9 |
| speed_auc | 0.817 | 0.818 | +0.001 (tied) | 21 |
| search_auc | 0.908 | **0.941** | +0.033 (much better) | 4 |
| accident_auc | 0.874 | **0.921** | +0.047 (much better) | 1 |
| injury_auc | 0.863 | **0.906** | +0.043 (much better) | 1 |

Composite (weighted sum) is -0.108 because the 0.25-weighted occurrence head took a hit AND 3D's late-epoch trajectory drifted down. But 4 of 7 heads got STRICTLY better, not the "uniformly worse" pattern a bad component would show. The per-head specialist ensembling at inference time would benefit from 3D's better speed_occ + search + accident + injury checkpoints.

**Refined recommendation**: don't add `--decorr` as a default flag, BUT re-evaluate per-head specialist ensembling — combine 3D's specialist checkpoints (speed_occ ep 18, search ep 4, accident/injury ep 1) with 3A's specialists where 3A wins (occ ep 18, trap ep 21, speed ep 23). Hybrid ensemble could outperform either alone.

This is a V3.2.0 deployment-time decision, not a "which loss to use during training" decision.

---

### Combined 3A+3D smoke (2026-04-25 ~21:30 — DONE)

| Variant | Composite | Best epoch | Δ vs baseline +0.726 |
|---|---|---|---|
| 3A alone (PLR) | +1.036 | 18 | +0.310 |
| **3A+3D combined** | **+0.975** | 25 | +0.249 |
| 3D alone | +0.618 | 9 | -0.108 |

Combined is **slightly worse than 3A-alone** by training-noise magnitude (-0.06). All per-head bests within ±0.001-0.002. Decorr trajectory: |corr| 0.24-0.28 throughout active window, never approaching 0.80 threshold.

**Conclusion**: 3D is a no-op confirmed. Decorr loss never engages because Phase 0C label partition keeps |corr| at 0.27.

Cmd: `MODELS_DIR=models/v3.1.1-3a-3d python next_gen/train.py --fold 4 --epochs 30 --use-plr --decorr` on desktop 4070 SUPER (~50 min after killing ComfyUI contention).

---

### 3A PLR fold 0 — the HARD fold (2026-04-25 ~21:30 — DONE)

Fold 0 is the early-data fold (sparse coverage, COVID-era regime shift). Vanilla v3.1.1 fold 0 was -0.21 composite — the only fold that LOST to XGBoost.

| Variant | Composite | Δ vs vanilla fold 0 (-0.21) |
|---|---|---|
| Vanilla v3.1.1 fold 0 | -0.210 | baseline |
| **3A PLR fold 0** | **+0.150** | **+0.360** |

**PLR fixes the hard fold** — and lifts it MORE than fold 4 (+0.360 vs +0.310). Fold 0 went from negative to positive composite.

Per-head fold 0 (PLR): occ_mae=0.431, speed_occ=0.037, trap=0.060, speed_auc=0.823, search=0.804, accident=0.698, injury=0.682. The rare-positive binary heads (search/accident/injury) still lag — that's the genuine difficulty of fold 0's sparse positive signal, not PLR's fault.

**Implication for v3.2.0**: confident PLR helps across the regime spectrum. Ship for the full 5-fold retrain.

Cmd: `MODELS_DIR=models/v3.1.1-3a-fold0 python next_gen/train.py --fold 0 --epochs 30 --use-plr` on laptop A4000.

---

### Phase 3 architecture decision matrix (FINAL — 2026-04-25 21:30)

| Component | Decision | Rationale |
|---|---|---|
| **3A PLR tokenizer** | ✅ **SHIP** | Δ +0.310 fold 4, +0.360 fold 0. Every per-head metric improves. 4× faster convergence. |
| **3D decorrelation loss** | ❌ **DROP from training** | No-op (|corr| already 0.27 due to Phase 0C label partition). Code stays in repo as reusable `cka_linear` + annealing utility. |
| 3E spatial GAT | ✅ **CODE READY (smoke pending)** | Per-batch wiring fix landed 2026-04-26 as `next_gen/encoders/spatial_cache.py` (Pattern 2 — per-epoch refresh + per-batch lookup). 19/19 unit tests pass; smoke ablation on fold 0 next. |
| 3B MoE Poisson heads | ⏸️ **DEFERRED** | Most complex. Don't need it: PLR alone closes most of the per-head gap. Reconsider if v3.2.0 still trails XGB on injury/accident at full 5-fold. |
| 3C head-attention masks | ⏸️ **DEFERRED** | Same reasoning as 3B. PLR did the job at the input layer; head-side attention masking is overkill. |

**v3.2.0 = v3.1.0 + PLR tokenizer**. Full 5-fold retrain expected to lift mean composite from +0.29 (v3.1.1) to ~+0.6-0.8 based on fold-4 +0.310 + fold-0 +0.360 data points. Worth the ~3 hr GPU time (laptop+desktop split).

3E discovery: existing wiring at `deep/model.py:433-434` is broken (per-batch call to a graph encoder that needs all-cells). See `next_gen/encoders/README.md` "Discovery — the existing 'activation' path is broken" for fix options. Recommended: per-epoch pre-compute (option 1) — ~3-5 day refactor, NOT a flag flip.

**Smoke ablation harness**: not yet built. Per `next_gen/BUILD_PROCESS.md` Step 5, each component needs a smoke run before shipping. Building the harness (a `next_gen/train.py` that wires `--use-plr` / `--decorr` etc. into the existing `train_multi.py` loop without modifying `deep/`) is the next infrastructure piece before any smoke runs.

## Next-gen direction — Phase 3 (5 components)

From the master plan. Implement together as one coordinated retrain (v3.2.0 or v4.0.0 if architecture changes warrant a major bump).

### 3A. PLR (Periodic-Linear) tokenizer

**Replaces** `NumericTokenizer` (`model.py:28-41`).

**What**: per-feature linear embedding gets a periodic component. Concat `[x, sin(2π · W·x), cos(2π · W·x)]` for learned frequency matrix W, then linear-project to `d_token`.

**Heaviest benefit**: cyclic features (hour, dow, lat, lng). Speeds up convergence + cleaner attention attributions.

**Expected lift**: +0.01-0.02 binary AUC + cleaner attention.

### 3B. Archetype-MoE Poisson heads

**Replaces** `OccurrenceHead × 3` (`model.py:388-390`).

**What**: each Poisson head gets 4 experts, top-2 gating conditioned on `(backbone_128, time_64)`. Each expert specializes on an enforcement regime (canyon trap / urban arterial / school zone / highway corridor).

**Expected lift**: speed_occurrence and trap heads decorrelated by routing. Structural fix complementing the Phase 0C label partition.

### 3C. Head-specific attention masks

**Adds** per-head L1-penalized gating matrix between backbone CLS attention and each task head's input.

**What**:
- Speed head attends to {road_maxspeed_*, road_curvature_mean, excessive/extreme_speed_frac, strictness_p75}
- Trap head attends to {stationary_trap_frac, laser_dominance, marked_frac, unmarked_radar_frac, canyon_speed_trap_score}
- Search head attends to {is_in_urban_core, dist_school_mean, vehicle attribute features}
- Accident head attends to {road_curvature_mean, weather features, crash history features}

**Why**: produces clean attention stories the UI can tell ("the speed head looked at X because it was constrained to"). Phase 1 attention extraction would show genuinely distinct fingerprints per head, not the current diffuse pattern.

### 3D. Soft decorrelation loss

**Adds** `λ × ReLU(|corr(speed_pred, trap_pred)| − 0.80)²` to training loss.

**Schedule**: λ annealed 0.05 → 0.15 over epochs 20-40.

**Why**: even with mutually-exclusive labels, model OUTPUTS may re-correlate. This caps it. Track via CKA between head representations each epoch; warn if CKA(speed, trap) > 0.90.

See [`08_loss_functions.md`](08_loss_functions.md) for implementation detail.

### 3E. Activate SpatialEncoder

**Already built** at `model.py:239-304`. Just needs `use_spatial=True` default in `train_multi.py`. ~10% epoch overhead. `h3_adjacency.npz` already on disk (1356 nodes, 6182 edges).

**Why**: sparse-data cells borrow signal from H3 neighbors. Particularly helpful for fold 0 (the hard regime).

---

## Phase 3 cost estimate

- **Modal**: 5 folds × ~75 min × L4 = ~6 GPU-hours @ ~$0.85/hr = ~$5
- **Local laptop+desktop split**: ~3 hours wall, $0
- **Dev**: 3-5 days for the 5 components + integration + smoke

---

## Phase 3 decision gate

- v3.2.0 composite vs v3.1.0 +0.29: should be ≥ +0.34 to justify the architectural complexity
- Per-head AUCs: at least 2 heads should close gap to XGB by ≥ 0.015
- CKA(speed_head, trap_head) drops from current ~0.95 to <0.70
- Attention fingerprints visibly distinct per head

If gates fail → Phase 3 components evaluated individually (which one dragged?) and reverted. If gates pass → v3.2.0 ships as new "v4.2-ensemble" toggle.

---

## References

- [FT-Transformer paper notes](../papers/tabular-transformers/README.md) — Gorishniy et al. 2021
- [MoE routing papers](../papers/moe-routing/README.md) — Switch Transformer, Mixtral, Soft MoE
- [Spatial GNN papers](../papers/temporal-spatial/README.md) — DCRNN, GraphWaveNet, GAT
- `archive/old-reports/ML_ARCHITECTURE_DIAGRAM.md` — full system diagram (predates Phase 3)
- `docs/strategy/hybrid-architecture-compositions-2026-04-24.md` — pattern explorations (Pattern 1 Hawkes, Pattern 2 MoE, Pattern 3 GAT/ST-GNN)
- `docs/strategy/ft-revision-plan-2026-04-24.md` — Phase 0/1/2 work that preceded Phase 3
- [`06_hyperparameters.md`](06_hyperparameters.md) — what changes per Phase 3 component
- [`13_ablation_studies.md`](13_ablation_studies.md) — slim 8-cell grid for Phase 3 component validation

---

## TODO

- [ ] Verify `d_token` value (code says 64, docs say 128 — pick one)
- [ ] Phase 3A: implement `PLRTokenizer` replacing `NumericTokenizer`
- [ ] Phase 3B: implement `ArchetypeMoEPoissonHead`
- [ ] Phase 3C: implement per-head attention masks with L1 penalty
- [ ] Phase 3D: integrate decorrelation loss into total_loss; add CKA tracking per epoch
- [ ] Phase 3E: flip `use_spatial` default to True; verify fold 0 lift
- [ ] Spell out v3.2.0 architecture diagram (companion to current `ML_ARCHITECTURE_DIAGRAM.md`)
