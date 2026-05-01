# Maryland Traffic-Stop Enforcement Risk Prediction

**Multi-task FT-Transformer with Periodic-Linear-ReLU tokenization**, applied to **14 years of Maryland public records (1,237,344 traffic stops, 2011-12-31 → 2026-04-21)**. Trained under walk-forward 5-fold cross-validation with 90-day embargo. Eight task heads sharing a single 128-dim backbone — three Poisson regression + five binary classifiers. Production deployment serving 331,475 cell-hour predictions at evasion.run/analytics.

This repository is the **academic carve-out of the Evasion V2 project** — the AI-analytics core of a larger system. Frontend, scanner pipeline, and surveillance-camera scrapers live in the parent monorepo and are intentionally excluded.

---

## How to evaluate this prototype

The prototype has two surfaces. **The deployed system is the running prototype**; this repository is the *training-and-inference pipeline that produced it*. An instructor or reviewer can engage with the work at three depths:

### 1. Read the executive report (no install required)

The full ~5,000-word executive report covering problem definition, originality, market analysis, viability, technical solution, and model performance results, with 7 figures and 48 references:

- **PDF (public, no login)**: https://evasion.run/academic-assets/executive-report.pdf
- **Architecture detail walkthrough (public)**: https://evasion.run/academic-assets/architecture-detail.html
- **Forward-pass animation (public)**: https://evasion.run/academic-assets/architecture-animation.html

### 2. Interact with the live prototype (account required)

The deployed system is the *running prototype*. A reviewer can interact with it directly via browser — no local install, no GPU required, no checkpoint download.

- **Live prediction map**: https://evasion.run/analytics — heatmap with hour/day-of-week selectors, model-version toggle (`v3` XGBoost ↔ `v4.3.1-ensemble` PLR FT-T), prediction-type toggle (total / speed / trap), per-cell popup with `[CLS]`-token similar-cell discovery, driver-archetype filter.
- **Methodology surface**: https://evasion.run/academic — full build-journey timeline, embedded architecture iframes, methodology cards, lessons-learned panel, results table.
- **Account**: sign up at https://evasion.run/signup, or contact the author for an evaluator account.

### 3. Run the code locally (for code-level review)

If the reviewer wants to inspect the pipeline at the code level:

```bash
git clone https://github.com/Jwessel2020/EvasionV2.git
cd EvasionV2/analytics-project
pip install -e .
python scripts/demo_architecture.py
```

The demo runs in **under 5 seconds on CPU** with synthetic torch.randn inputs, producing output for all 8 heads. It demonstrates the architecture is correctly wired without requiring any data download or trained checkpoint. Expected output:

```
Instantiated UnifiedModel:
  d_token=128  n_layers=4  n_heads=8  k=8 PLR freqs
  trainable params: ~1M

PLR tokenizer:
  input shape:  (16, 85)
  output shape: (16, 85, 128)

Forward pass on a batch of 16 synthetic stops:
  3 Poisson heads:
    occurrence          shape=(16)  exp(log_rate).mean()=...
    speed_occurrence    shape=(16)  exp(log_rate).mean()=...
    trap                shape=(16)  exp(log_rate).mean()=...
  5 binary heads:
    speed/search/accident/injury/disposition  ...

  All 8 heads produced output. Architecture is wired correctly.
```

---

## Headline results

Per-head performance: XGBoost Optuna-tuned baseline vs FT-Transformer 5-fold ensemble (v3.3.1, training-best per fold averaged). Composite metric ≡ `Σ_Poisson (xgb_mae − pred_mae)/xgb_mae + Σ_binary (pred_auc − xgb_auc)`. The result is **heterogeneous, honestly reported**: FT-T wins decisively on the three Poisson count heads and on disposition, ties on speed-related, and underperforms XGBoost on the three rare-event binary classifiers (search / accident / injury).

| Head | Metric | XGBoost | FT-T 5-fold ensemble | Δ |
|---|---|---|---|---|
| Occurrence count | MAE (lower=better) | 0.339 | **0.352** | −3.8% (slight regression) |
| Speed-occurrence count | MAE | 0.086 | **0.030** | **+65.1%** |
| Speed-trap count | MAE | 0.091 | **0.050** | **+45.1%** |
| Speed-related | AUC (higher=better) | 0.842 | **0.867** | +0.025 |
| Search conducted | AUC | 0.883 | 0.844 | −0.039 (FT-T worse) |
| Accident | AUC | 0.894 | 0.783 | −0.111 (FT-T worse) |
| Personal injury | AUC | 0.881 | 0.753 | −0.128 (FT-T worse) |
| Citation disposition | AUC | 0.650* | **0.745** | +0.095 (\*XGB estimate; head new in v3.3.0) |
| **Composite (5-fold mean)** | | **baseline (=0)** | **+0.913** | net positive |

The composite gain decomposes as **+1.064 from the three Poisson count heads** (dominated by speed-occurrence and trap MAE drops of 65% and 45%) plus **−0.158 from the five binary heads** (where speed-related and disposition wins are partially offset by search / accident / injury losses). FT-T excels at fine-grained spatiotemporal count prediction; XGBoost's per-head Optuna tuning extracts more signal per parameter on rare-event binary classification. The +0.913 composite is real but heterogeneous.

Architectural ablations measured against the PLR-equipped baseline (composite Δ; one earned ship, four did not):

- **PLR tokenizer** (Phase 3A): **+0.310 → shipped in v3.2.0**
- Spatial GAT cache (Pattern 2): −0.156 → rejected
- LSTM late-fusion (Pattern 1): −0.067 → rejected
- Hawkes self-excitation (Pattern 5): −0.030 → rejected
- MoE routing heads (Pattern 4): −0.226 → rejected

Full per-fold metrics in [`results/baseline-comparison.json`](results/baseline-comparison.json) and [`results/v3.3.1-per-fold-metrics.json`](results/v3.3.1-per-fold-metrics.json).

---

## Architecture

```
85 numeric features ──► PLR Tokenizer (k=8 freqs)  ──┐
                        d_token=128                   │
                                                      ├─► FeatureTokenizer ──► 4× Transformer Block
6 categorical features ─► nn.Embedding(N, 128) ───────┤   + [CLS] token         8 heads × ReGLU FFN
                                                      │   ~110 tokens          pre-LayerNorm
24 runtime TE columns ──► Linear → 128-d ─────────────┤                        attn dropout 0.2
                                                      │                                 │
~80 cell features ──────► MLP → 128-d ────────────────┘                                 │
                                                                                        ▼
                                                                                 [CLS] readout
                                                                                  128-d shared
                                                                                  representation
                                                                                        │
                                              ┌──────────────────────────┬──────────────┴──────────────┐
                                              ▼                          ▼                             ▼
                                       3 Poisson heads             5 binary heads             (per-head specialist
                                       (log-rate + NLL)            (sigmoid + BCE)              checkpoints save the
                                       occurrence                  speed_related                best epoch per head;
                                       speed_occurrence            search_conducted             ensemble loads each
                                       trap                        accident                     head's specialist)
                                                                   personal_injury
                                                                   disposition

Total params: ~1.14M. Hyperparameters: d_token=128, n_layers=4, n_heads=8, n_periodic_freqs=8.
leak_safe=True drops is_post_covid + is_covid_era per the std-floor leak fix.
```

Interactive walkthrough: [`docs/architecture-detail.html`](docs/architecture-detail.html) (open in browser). Forward-pass animation: [`docs/architecture-animation.html`](docs/architecture-animation.html).

---

## Methodology highlights

The training methodology — not the architecture itself — carries the bulk of the technical contribution. Five points:

1. **Walk-forward 5-fold CV with 90-day embargo** (`src/data/cross_validation.py`). Each fold's training data is strictly chronological and earlier than its test slice. Random k-fold leaks future information into past folds and was the underlying cause of multiple leakage pathways. Embargo concept from quantitative-finance ML (López de Prado 2018).
2. **Five-stage data-leak audit**. Five distinct leakage pathways were caught and fixed during development: (A) pre-computed target-encoded columns, (B) saturated-sigmoid std-floor bug producing AUC=0.5000 across 4 heads, (C) pre-split rolling features + train+val encoder fit, (D) per-cell `citation_rate` aggregation, (E) `violation_type` 4-tuple target-encoded recovery (the leak hiding behind D). See [`docs/methodology/04_cv_strategy.md`](docs/methodology/04_cv_strategy.md) and the full §3 of the executive report for the audit table with associated commit hashes.
3. **Periodic-Linear-ReLU (PLR) numeric tokenization**. Each numeric *x* expanded as $[x, \sin(2\pi W_1 x), \cos(2\pi W_1 x), \dots, \sin(2\pi W_8 x), \cos(2\pi W_8 x)]$ → Linear → ReLU → 128-dim token. Captures cyclic structure (hour-of-week 168 rates, day-of-year, longitude wraparound) that linear embeddings miss. Code in [`src/architecture/plr_tokenizer.py`](src/architecture/plr_tokenizer.py).
4. **Multi-task with shared backbone** (`src/architecture/unified_model.py`). Eight task-specific MLP heads share a single 128-dim `[CLS]` readout; backbone amortization reduces per-head sample requirements and acts as implicit regularization.
5. **Per-head specialist checkpoints**. Binary heads peak at epochs 2-3; Poisson heads peak at epochs 18-25. Saving 9 `.pt` files per fold (1 last + 8 per-head specialists) and loading the best checkpoint per head produces ~+0.08 composite improvement over a single-checkpoint ensemble.

---

## Repository layout

```
analytics-project/
├── README.md                       # this file
├── EXPORTING.md                    # subtree-split + cleanup checklist
├── pyproject.toml                  # editable install + deps
├── requirements.txt                # pinned dependency versions
│
├── src/                            # the package — `pip install -e .` then `from src.X import Y`
│   ├── config.py                   # constants, paths, CV parameters
│   ├── architecture/
│   │   ├── unified_model.py        # FT-T backbone + 8 task heads (the core class)
│   │   └── plr_tokenizer.py        # Periodic-Linear-ReLU numeric tokenizer
│   ├── data/
│   │   ├── _encoders.py            # FoldSafeTargetEncoder (Bayesian smoothing s=20)
│   │   ├── feature_engineering.py  # Phase 0A-0E feature build (197 cols)
│   │   ├── cross_validation.py     # walk-forward 5-fold splitter + 90-day embargo
│   │   └── datasets.py             # PyTorch Datasets + std-floor leak guard
│   ├── training/
│   │   └── train.py                # multi-task training loop, per-head specialist checkpoints
│   └── evaluation/
│       └── inference.py            # 5-fold ensemble inference (multi-version checkpoint loader)
│
├── scripts/
│   └── demo_architecture.py        # 5-second forward-pass demo, no download needed
│
├── docs/
│   ├── architecture-detail.html    # interactive design walkthrough
│   ├── architecture-animation.html # forward-pass animation
│   └── methodology/                # 5 methodology docs (CV, architecture, ablations, loss, GNN/LSTM)
│
├── results/
│   ├── baseline-comparison.json    # XGBoost vs FT-T per-fold metrics (the headline numbers)
│   └── v3.3.1-per-fold-metrics.json  # full per-fold per-head metrics
│
└── checkpoints/                    # populated via `scripts/download_checkpoint.sh` (see below)
```

**Training and inference are separated**: `src/training/train.py` and `src/evaluation/inference.py` are independent entry points. Training produces per-fold per-head `.pt` checkpoints in `checkpoints/`; inference loads them and writes `data/{stop,bin}_predictions_v4_ensemble.parquet`.

---

## Reproducing predictions

The full reproduction path requires the preprocessed feature parquet (~737 MB) and at least one fold checkpoint (~50 MB per fold; 9 per fold for full ensemble inference). These are too large for git and live in GitHub Releases.

```bash
# 1. (one-time) Editable install — makes `src.X` imports work
pip install -e .

# 2. Smoke check: architecture is correctly wired (no data needed)
python scripts/demo_architecture.py

# 3. (Optional) Download v3.3.1 release artifacts — sample data + 1-fold checkpoint
#    (Replace <release-tag> with the latest tag from the GitHub Releases page.)
mkdir -p data_samples checkpoints
curl -L https://github.com/Jwessel2020/EvasionV2/releases/download/<release-tag>/stops_features_sample.parquet \
     -o data_samples/stops_features_sample.parquet
curl -L https://github.com/Jwessel2020/EvasionV2/releases/download/<release-tag>/unified_phase2_fold0.pt \
     -o checkpoints/unified_phase2_fold0.pt

# 4. Run inference on the sample (single fold for speed). CUDA required.
MODELS_DIR=./checkpoints \
  PRED_PATH=./data_samples/stops_features_sample.parquet \
  python -m src.evaluation.inference --folds 0 --batch-size 256

# Output: data/stop_predictions_v4_ensemble.parquet (per-stop probabilities)
#         data/bin_predictions_v4_ensemble.parquet (per-cell-hour Poisson rates)
```

**CUDA required for real inference.** A 1.14M-parameter FT-Transformer with PLR tokenizer over 110-token sequences is fundamentally a GPU workload at any meaningful batch size. The `--cpu` flag exists in the inference script for debugging but in practice CPU inference on even the 10K-row sample runs into hours rather than minutes. The 5-second `scripts/demo_architecture.py` demo is a 16-row synthetic forward pass and is the only path that runs comfortably on CPU. For full inference you need a recent NVIDIA GPU with CUDA 11.8+ or 12.x and at least ~2 GB free VRAM per fold.

Training from scratch (60 min on a single A4000 / RTX 4070 GPU per fold; CPU is impractical):

```bash
python -m src.training.train --fold 0 --epochs 60 --use-plr
```

Outputs: 9 checkpoints per fold (`unified_phase2_fold0.pt` plus 8 per-head specialists like `unified_phase2_fold0_best_speed_auc.pt`). The full 5-fold ensemble takes ~5 hours on a single GPU; the v3.3.1 production retrain ran in 108 minutes wall time on Modal Cloud (4× L4 GPUs in parallel via `.starmap()`, $4.36 spend per retrain).

---

## Environment requirements

- **Python**: 3.11 or 3.12 (uses PEP 604 typing syntax)
- **PyTorch**: 2.1+ with CUDA. CPU works only for the 5-second `demo_architecture.py` synthetic forward pass. Real inference and training both require an NVIDIA GPU with CUDA 11.8+ or 12.x and ~2 GB free VRAM per fold. There is no usable CPU path for inference at the dataset scale (1.24M stops or even the 10K sample).
- **OS**: Linux, macOS, or Windows (production trains on Windows + WSL2; Modal Cloud uses Linux containers)
- **RAM**: 8 GB minimum for inference on the sample; 16 GB+ recommended for training; 32 GB+ recommended for full-dataset feature engineering on the 1.24M-row Maryland parquet
- **Disk**: ~250 MB for the standalone repo; +750 MB for the full feature parquet; +5 GB for the full v3.3.1 ensemble (5 folds × 9 checkpoints × ~50 MB each)

Install via `pip install -e .` (uses `pyproject.toml`) or `pip install -r requirements.txt` (pinned pip deps). The `[dev]` extra adds pytest, ruff, mypy.

---

## References

The full reference list is in the executive report (48 entries). Highlights:

- **Gorishniy et al. NeurIPS 2021** — *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer). arXiv:2106.11959.
- **Gorishniy et al. NeurIPS 2022** — *On Embeddings for Numerical Features in Tabular Deep Learning* (PLR). arXiv:2203.05556.
- **Gorishniy et al. ICLR 2024** — *TabR: Tabular Deep Learning Meets Nearest Neighbors*. arXiv:2307.14338.
- **Kazemi et al. 2019** — *Time2Vec*. arXiv:1907.05321.
- **Sahr, White, Kimerling 2003** — *Geodesic Discrete Global Grid Systems*. CGIS 30(2):121-134. (H3 theoretical foundation.)
- **Caruana 1997** — *Multitask Learning*. Machine Learning 28(1):41-75.
- **Sener & Koltun NeurIPS 2018** — *Multi-Task Learning as Multi-Objective Optimization*. arXiv:1810.04650.
- **Pargent et al. 2022** — *Regularized target encoding outperforms traditional methods*. Computational Statistics 37:2671-2692.
- **Bergmeir, Hyndman, Koo 2018** — *A note on the validity of cross-validation for evaluating autoregressive time series prediction*. CSDA 120:70-83.
- **López de Prado 2018** — *Advances in Financial Machine Learning*. Wiley. (Embargo concept origin.)
- **Pierson et al. Nature Human Behaviour 2020** — *A large-scale analysis of racial disparities in police stops across the United States* (Stanford Open Policing, 95M+ stops).
- **Mohler et al. JASA 2011** — *Self-Exciting Point Process Modeling of Crime* (the rejected Hawkes pattern).

Data sources:
- **Maryland Open Data Portal** — https://imap.maryland.gov/ (1.24M traffic-stop labels)
- **U.S. Census TIGER/Line** — https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html (national road geometry, 5.8M-row fuzzy index)
- **FHWA HPMS** — https://www.fhwa.dot.gov/policyinformation/hpms.cfm (Annual Average Daily Traffic per-segment volume)
- **NOAA Hourly Observations**, **U.S. Census Bureau ACS**, **OpenStreetMap**, **MD iMAP traffic cameras**, **FARS / state crash records**.

---

## Citing

If this work is useful in your research, please cite as:

```bibtex
@misc{evasionv2_2026,
  author = {Calvo, Stan and Drake Salda{\~n}a, Miguel and Garc{\'i}a-Maroto del R{\'i}o, Julio
            and Juzgado Garc{\'i}a-Aranda, Juan and Wessel, Julian Lloyd},
  title = {Periodic-Linear-ReLU Feature Tokenization for Police Enforcement Prediction:
           A 14-Year Maryland Study with Five-Stage Leak Auditing},
  year = {2026},
  url = {https://evasion.run/academic-assets/executive-report.pdf},
  note = {Final Project, Deep Learning Application course}
}
```

---

## Acknowledgments

This is the academic carve-out of the [Evasion V2](https://evasion.run) project. The full system (frontend, scanner pipeline, Flock + ALPR camera infrastructure, 51K+ camera nationwide layer) remains in the parent monorepo and is not part of this repository. The /academic page on the live site is the human-readable counterpart to this README; the executive report PDF carries the complete technical and policy analysis.

For reproduction questions or evaluator-account requests, contact the author.
