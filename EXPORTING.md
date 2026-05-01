# Exporting `analytics-project/` as its own GitHub repository

This directory was carved out of the parent Evasion V2 monorepo as the
self-contained academic version. To publish it as its own GitHub repo,
two paths.

---

## Option A — `git subtree split` (preserves commit history)

From the parent monorepo root:

```bash
# 1. Split the analytics-project/ subdirectory into its own branch.
#    Commits that touched files in analytics-project/ become the new history.
git subtree split --prefix=analytics-project --branch=academic-export

# 2. Push the academic-export branch to a fresh GitHub repo as its main:
git push <new-repo-url> academic-export:main

# 3. (optional) Clean up the export branch in the monorepo
git branch -D academic-export
```

The new repo will have only the analytics-project history. The parent
monorepo continues to have analytics-project/ as a regular subdirectory.

---

## Option B — fresh-init (clean history, simpler)

```bash
# 1. Copy the directory out of the monorepo:
cp -r analytics-project /tmp/ai-traffic-analytics
cd /tmp/ai-traffic-analytics

# 2. Initialize a fresh git repo:
git init
git add .
git commit -m "Initial extract from Evasion V2 monorepo"

# 3. Push to a fresh GitHub repo:
git remote add origin <new-repo-url>
git push -u origin main
```

Loses commit history but produces a clean single-commit start. Recommended
for academic submission since the development history in the parent
monorepo includes scanner / surveillance commits that aren't part of the
academic story.

---

## What's intentionally NOT in the export

- **Frontend code** (`src/` Next.js application) — proprietary product
- **Scanner pipeline** (`ml/transcription`, `ml/extraction`, `ml/geocoding`) — surveillance
- **Camera scrapers** (`cameras/`) — surveillance
- **Production database schema** (`prisma/`) — has user-tracking tables
- **Worker compute donation infrastructure** (`worker/`)
- **Deploy automation** (`.github/workflows/`, `deploy/`, `docker-compose.yml`)
- **Full traffic-violation dataset** (737 MB, public-records source) — instructor
  reproducibility uses a sample (see `data_samples/` if shipped) or downloads
  fresh from https://imap.maryland.gov/

---

## Per user direction

No `LICENSE` file is included. Rights remain with the author; this is an
academic showcase repository, not a public open-source release.

---

## Standalone-runnable status (May 2026)

The cleanup that was previously deferred has now been applied — this directory is **standalone-runnable as of commit `33140fe` onward**:

- ✅ All `import config` calls now use `from src import config` (slim local copy at `src/config.py`)
- ✅ `from deep.data import X` → `from src.data.datasets import X`
- ✅ `from deep.model import UnifiedModel` → `from src.architecture.unified_model import UnifiedModel`
- ✅ `from _encoders import FoldSafeTargetEncoder` → `from src.data._encoders import FoldSafeTargetEncoder`
- ✅ `pyproject.toml` for `pip install -e .`
- ✅ `scripts/demo_architecture.py` runs end-to-end on synthetic data with no external dependencies (5 seconds, CPU)
- ✅ Inference path is parquet-in / parquet-out (no DB upload — DB integration lives in the parent monorepo)
- ✅ Sample data + checkpoints documented as GitHub Release artifacts (avoid bundling 737 MB feature parquet + 250 MB ensemble checkpoints in git)

To verify: `cd analytics-project && pip install -e . && python scripts/demo_architecture.py`
