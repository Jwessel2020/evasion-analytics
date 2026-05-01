# Loss Functions

**Drafted 2026-04-25.** Poisson NLL for the 3 occurrence heads, BCE-with-logits + pos_weight for the 4 binary heads, weighted sum aggregation, planned Phase 3D soft decorrelation loss.

---

## Purpose

Loss design encodes what we tell the model to optimize. Two questions matter most:

1. **Per-head**: is the chosen loss appropriate for the target distribution? (Poisson assumes Poisson — but enforcement counts may be overdispersed; pos_weight balances class imbalance — but maybe focal loss does it better for the rare injury head.)
2. **Aggregate**: how do we combine 7 losses into one scalar gradient signal? Weighted sum is convenient but not Pareto-respecting.

---

## Current state (v3.1.0)

### Per-head losses

| Head | Loss function | Where | Notes |
|---|---|---|---|
| `occurrence` | `F.poisson_nll_loss(log_input=True)` | `train_multi.py:321-360` | Expects pre-exponentiated logit |
| `speed_occurrence` | same | same | |
| `trap` | same | same | |
| `is_speed_related` | `F.binary_cross_entropy_with_logits(pos_weight=pw)` | `train_multi.py:321-360` | pw = min(neg/pos, 50) |
| `search_conducted` | same | same | |
| `accident` | same | same | |
| `personal_injury` | same | same | pw saturates near cap (47.0) |

### Poisson NLL detail

`F.poisson_nll_loss(log_input=True, full=False, reduction='mean')`:
- Expects `input` as `log(λ)` (i.e., model outputs the logit, exp happens inside loss)
- Computes `-target × input + exp(input) + log(target!)` averaged over batch
- `full=False` skips the Stirling-approx `log(target!)` term (constant w.r.t. params, fine for gradient)

### BCE detail

`F.binary_cross_entropy_with_logits(pos_weight=pw)`:
- Expects `input` as logit (sigmoid happens inside)
- Computes `pos_weight × y × log(σ(x)) + (1-y) × log(1-σ(x))` (negated, summed, averaged)
- Per-class `pos_weight` upweights positive examples to compensate for class imbalance

### Aggregation

```python
loss_poisson = 0.25 * L_occ + 0.10 * L_speed_occ + 0.05 * L_trap
loss_binary  = 0.10 * L_speed + 0.15 * L_search + 0.20 * L_acc + 0.15 * L_inj
total_loss   = loss_poisson + loss_binary  # via separate backward calls
```

Sum-of-weights = 1.00. **Two backward calls** in alternating multi-task pattern (`train_multi.py:300-450`); gradients accumulate before optimizer step.

### pos_weight derivation

```python
pos_count = batch_y.sum()
neg_count = batch_y.numel() - pos_count
raw_pw = neg_count / max(pos_count, 1)
pw = min(raw_pw, 50.0)  # cap to prevent explosion
```

Computed per-batch (so pw can vary slightly batch-to-batch with rare positives), capped at 50 to stop gradient explosion on personal_injury.

### Grad scaling + clipping

- `GradScaler` (AMP) wraps both backward calls
- After both backwards: `unscale_` → `clip_grad_norm_(model.parameters(), max_norm=1.0)` → `scaler.step` → `scaler.update`

---

## Known gaps / pain points

- **No focal loss** / hard-example up-weighting. Personal-injury (1.7%) head is a candidate — focal might dominate pos_weight at that imbalance.
- **No per-head gradient scaling**. `clip_grad_norm_` at 1.0 applies globally. If one head's gradient is much larger than another's, the dominant head effectively sets direction.
- **Poisson assumes Poisson distribution** (mean = variance). Enforcement counts may be overdispersed (some cells extremely high variance, like rush-hour I-95). Negative Binomial could be a better fit.
- **No zero-inflated Poisson option** for the trap head. Many cells have zero traps for almost all hour slots — zero-inflated Poisson explicitly models the structural-zero process separately.
- **Loss weights set empirically once** (rebalanced 2026-04-24). No grid search.
- **Class imbalance handled only at loss level** — no oversampling or focal sampling at data level.

---

## Open questions

- Is enforcement count overdispersed? Can compute `Var(stop_count)/Mean(stop_count)` on training bins; if >> 1, Poisson is wrong.
- Should personal_injury get focal loss (γ=2) instead of pos_weight=50?
- Should we use loss-balancing methods like GradNorm or PCGrad to dynamically weight heads instead of static weights?
- Should the trap head be zero-inflated Poisson?

---

## Next-gen direction — Phase 3D + 3B + Phase 5 Hawkes

### 3D: Soft decorrelation loss

```python
# Per batch, AFTER computing per-head predictions
speed_pred = sigmoid(out["speed"]).flatten()
trap_pred  = exp(out["trap"]).flatten()  # Poisson rate

# Pearson correlation
corr = pearson_corr(speed_pred, trap_pred)
decorr_loss = lambda_decorr * F.relu(abs(corr) - 0.80) ** 2

total_loss = total_loss + decorr_loss
```

`lambda_decorr` annealed 0.05 → 0.15 over epochs 20-40. Why annealing: at epoch 0, predictions are ~uniform; correlation is undefined. Wait until model has structure before penalizing redundancy.

Track `corr` per epoch in metrics.json. Warn if `corr > 0.95` even with the loss active (means the loss isn't biting).

### 3B: MoE load-balancing loss

When MoE Poisson heads ship (Phase 3B), add load-balancing loss to prevent expert collapse:

```python
load_balance_loss = lambda_balance * (mean_per_expert(routing_prob) * mean_per_expert(routing_decision)).sum()
```

`lambda_balance = 0.01`. Standard MoE practice (Switch Transformer paper).

### Phase 5: Hawkes baseline + neural residual loss

**Status (2026-04-26)**: code shipped on branch `next-gen/p5-hawkes`
(`ml/research/next_gen/heads/hawkes.py` + tests + `hawkes_train_hooks.py`
sidecar wiring). NOT YET smoke-tested. See
`docs/next-gen/planning/16_lstm_gnn_integration.md` Pattern 5 +
`docs/exploration/hawkes-implementation-plan-2026-04-24.md` for the
full design rationale (Q1-Q5: variant selection, kernel form,
numerical gotchas, production-deployment evidence, integration plan).

The `occurrence` Poisson loss is augmented (NOT replaced) with a
parallel Hawkes-augmented occurrence loop:

```python
# Per batch from a TemporalOccurrenceDatasetWithCellIdx loader (date-
# specific (cell, hour) samples with 168-h count history per row):
neural_resid = model(...)["occurrence"]     # the existing OccurrenceHead
λ_hawkes    = hawkes_head(cell_idx, cell_sequence)  # closed-form ETAS sum
log_rate    = log(λ_hawkes + softplus(neural_resid) + 1e-8)
loss_haw    = F.poisson_nll_loss(log_rate, target, log_input=True)

total_loss = loss_poisson  +  args.hawkes_loss_weight * loss_haw
                              # default: 0.25 = LOSS_WEIGHTS["occurrence"]
```

**Why a parallel loop** (not an in-place replacement of the bin-based
occurrence loss):

The bin-based `OccurrenceDataset` operates on (cell, hour, day-of-week)
weekly templates and emits 3 targets (total / speed / trap) per bin.
The Hawkes head needs date-specific (cell, hour) samples with a
168-hour history window — `TemporalOccurrenceDataset` provides those
for the total-count target only. By keeping both loops:

- speed_occurrence + trap heads stay trained (bin loop, weekly templates)
- occurrence head specializes via the Hawkes residual (temporal loop,
  date-specific) AND continues to learn the weekly template (bin loop)

The dual-signal training gives the OccurrenceHead the broader
temporal grounding from bins plus the residual sharpening from the
Hawkes-augmented date-specific samples.

**Loss-weight choice**: `args.hawkes_loss_weight = 0.25` mirrors
`LOSS_WEIGHTS["occurrence"] = 0.25` so the Hawkes pass is roughly
balanced against the bin pass. Higher weights (>0.5) risk
over-fitting the residual on the 500K temporal samples; lower weights
(<0.1) likely under-trains the Hawkes residual. Smoke-test the
default first; adjust only if `hawkes_nll` plateaus high vs the bin
`occ_loss`.

**Hyperparameters** (CLI flags on `next_gen/train.py`):

| Flag | Default | Why |
|---|---|---|
| `--hawkes-alpha-init` | 0.3 | Mohler LAPD trial: branching ratios in [0.3, 0.7] for crime |
| `--hawkes-beta-init` | 0.115 | log(2)/6h half-life — consensus crime self-excitation timescale |
| `--hawkes-beta-min` | 0.01 | Half-life ceiling ≈ 69h (~3 days) |
| `--hawkes-beta-max` | 1.0 | Half-life floor ≈ 40min |
| `--hawkes-loss-weight` | 0.25 | Matches LOSS_WEIGHTS["occurrence"] |
| `mu_init` (per-cell) | empirical Bayes shrunk | n_c · MLE_c + τ · global_μ ÷ (n_c + τ); τ = max(50, median(n_c)/5) |

All three params (α, β, μ_c) are unconstrained-real `nn.Parameter`s
passed through `softplus` (and clamp for β) on each forward — gradient-
stable by construction.

**Numerical stability**:
- β clamped to [0.01, 1.0]: half-life range [0.69h, 69h] covers the
  Mohler-style crime-self-excitation regime with headroom.
- Excitation sum computed via `logsumexp(log(count) - β·lag)` instead
  of direct `Σ count · exp(-β·lag)` — avoids exp-underflow for large
  lag · β products. Count==0 rows masked via `log_count = -1e9`
  sentinel (logsumexp ignores them cleanly).
- Test coverage: `test_finite_at_beta_min`, `test_finite_at_beta_max`,
  `test_finite_for_all_zero_history`, `test_beta_clamp_effective`
  (all in `next_gen/tests/test_hawkes.py`).

**References**:
- Mohler et al. 2015 (JASA) — LAPD ETAS RCT (7.4% crime reduction)
- Zhang et al. 2024 (CL-ETAS, GJI) — earthquake forecasting with Hawkes + ConvLSTM residual
- Steven Morse Hawkes notes — closed-form recurrence for exponential kernel
- `tick.HawkesExpKern` — production parametric Hawkes (used as v2-upgrade reference, not the v1 path)

### Optional: focal loss for injury

```python
gamma = 2.0
p_t = sigmoid(out["injury"])  # for positive samples
focal_weight = (1 - p_t) ** gamma  # downweights easy positives
L_inj_focal = -focal_weight * y * log(p_t) - (1-y) * log(1-p_t)
```

Worth A/B vs current pos_weight=50 — may improve injury AUC by 0.005-0.015.

---

## References

- [`03_label_design.md`](03_label_design.md) — what each loss is fitting
- [`05_model_architecture.md`](05_model_architecture.md) — Phase 3D + 3B context
- [`06_hyperparameters.md`](06_hyperparameters.md) — loss-weight values + pos_weight cap
- [`07_training_loop.md`](07_training_loop.md) — how losses feed into the loop
- `docs/strategy/hybrid-architecture-compositions-2026-04-24.md` — MoE pattern (Pattern 2)
- [Switch Transformer paper notes](../papers/moe-routing/README.md) — load-balancing loss derivation

---

## TODO

- [ ] Compute Var/Mean ratio on training bins → decide Poisson vs Negative Binomial
- [ ] Implement focal loss option for injury head; A/B vs pos_weight=50
- [ ] Implement Phase 3D decorrelation loss with annealing
- [ ] Implement Phase 3B MoE load-balancing loss
- [ ] Add per-head loss + gradient-norm logging to `metrics.json`
