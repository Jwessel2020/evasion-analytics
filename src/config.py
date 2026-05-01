"""Project-wide configuration for the analytics-project standalone repository.

Embedded copy of the constants from the parent monorepo's `ml/research/config.py`,
trimmed to what the standalone training + inference paths actually need. No
database URLs, no scanner-pipeline paths — those live in the parent monorepo.

Override the model checkpoint directory via the `MODELS_DIR` environment
variable (e.g. when running with v3.2.0 vs v3.3.1 weights without clobbering
either set):

    MODELS_DIR=./checkpoints/v3.3.1 python -m src.evaluation.inference --folds 0
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root — `src/config.py` is one level below the analytics-project root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_SAMPLES_DIR = REPO_ROOT / "data_samples"
RESULTS_DIR = REPO_ROOT / "results"

# Checkpoint directory. Defaults to checkpoints/ but can be overridden so a
# user can point at a downloaded release tarball without moving files.
_models_override = os.environ.get("MODELS_DIR")
MODELS_DIR = Path(_models_override) if _models_override else (REPO_ROOT / "checkpoints")

# Default model version tag (used by inference path on multi-version repos).
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v3.3.1")

# ---------- Cross-validation parameters ----------

CV_RANDOM_STATE = 42
WALK_FORWARD_START = "2018-01-01"
WALK_FORWARD_TRAIN_MIN_MONTHS = 24
WALK_FORWARD_VAL_MONTHS = 6  # used as the embargo window in walk-forward
WALK_FORWARD_TEST_MONTHS = 6
WALK_FORWARD_N_SPLITS = 5

# ---------- Spatial / feature engineering ----------

H3_RESOLUTION = 9  # ~150m hexagonal cells

POI_THRESHOLDS = {
    "school_zone_m": 300,
    "near_hospital_m": 500,
    "near_police_m": 1000,
    "near_courthouse_m": 1000,
    "urban_core_m": 500,
}
ROLLING_WINDOWS = [7, 30, 90]
ROAD_JOIN_SEARCH_RADIUS_M = 200

QC_MAX_SNAP_DISTANCE_M = 50
QC_MAX_SPEED_OVER = 80
QC_MIN_VEHICLE_YEAR = 1960
QC_MAX_VEHICLE_YEAR = 2026

# ---------- Outcome targets ----------

OUTCOME_TARGETS = [
    "search_conducted",
    "personal_injury",
    "accident",
    "alcohol",
    "is_speed_related",
]
OCCURRENCE_TARGET = "stop_count"

# ---------- Paths to expected parquet artifacts (gitignored at production scale) ----------

PATH_STOPS_FEATURES = DATA_DIR / "stops_features.parquet"
PATH_FOLDS = DATA_DIR / "train_test_splits.parquet"

# Sample-size paths shipped with the repo so an instructor can run inference
# end-to-end without downloading the full 737 MB feature parquet.
PATH_STOPS_FEATURES_SAMPLE = DATA_SAMPLES_DIR / "stops_features_sample.parquet"
PATH_FOLDS_SAMPLE = DATA_SAMPLES_DIR / "train_test_splits_sample.parquet"


def ensure_dirs() -> None:
    """Create output directories on import (idempotent)."""
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
