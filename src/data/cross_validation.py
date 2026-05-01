#!/usr/bin/env python3
"""Stage 04 — Build CV folds (walk-forward temporal + spatial holdout).

Produces a `data/train_test_splits.parquet` file that maps each row of
`stops_features.parquet` to fold assignments for every CV strategy:

    - WalkForwardSplit (primary): expanding training window with a fixed
      validation + test window after it
    - LeaveCountyOutSplit (secondary): holds out N counties, tests spatial
      generalization
    - CombinedSplit (tertiary): walk-forward AND spatial holdout

Downstream training scripts read this parquet to slice X/y by fold index,
guaranteeing every experiment uses the SAME splits.

Usage:
    cd ml/research
    python pipelines/04_build_folds.py
    python pipelines/04_build_folds.py --strategy temporal
    python pipelines/04_build_folds.py --strategy spatial

Output:
    data/train_test_splits.parquet  (one row per stop × strategy × fold)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from src import config


# ---------- Walk-forward temporal splitter ----------

class WalkForwardSplit(BaseCrossValidator):
    """Walk-forward cross-validation for time-ordered data.

    For each fold, training data is everything up to `train_end`, validation
    is the `val_months`-wide window after that, and test is the `test_months`
    window after validation. The training window EXPANDS each fold (includes
    all prior data); validation and test windows are fixed-size and slide
    forward.

    This guarantees:
      - No future data leaks into the training set
      - Validation is used for model selection / early stopping
      - Test is touched only once per fold, for final metrics

    Parameters
    ----------
    date_col : str
        Name of the column holding the stop_date (as datetime64).
    start_date : str
        Earliest date in the data (training begins here).
    train_min_months : int
        Minimum size of the initial training window.
    val_months : int
        Width of the validation window (held out from training, used for
        hyperparameter selection + early stopping).
    test_months : int
        Width of the test window (held out completely until final eval).
    n_splits : int
        Number of walk-forward folds.
    """

    def __init__(
        self,
        date_col: str = "stop_date",
        start_date: str = "2018-01-01",
        train_min_months: int = 24,
        val_months: int = 6,
        test_months: int = 6,
        n_splits: int = 5,
        embargo_days: int = 90,
    ):
        # `embargo_days` shifts the train_end cutoff back by this many days
        # to absorb rolling-history features (`stops_last_7d/30d/90d`,
        # `days_since_last_stop_here`, `csv_era_stops`) that compute on
        # the full dataset before fold split. Without this, the 90-day
        # lookback at test_start pulls counts from train rows.
        # Default 90 = max(config.ROLLING_WINDOWS).
        # See docs/walk-forward-cv-audit-2026-04-24.md.
        self.date_col = date_col
        self.start_date = pd.Timestamp(start_date)
        self.train_min_months = train_min_months
        self.val_months = val_months
        self.test_months = test_months
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterable[
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Yield (train_idx, val_idx, test_idx) triples."""
        if self.date_col not in X.columns:
            raise ValueError(f"{self.date_col} not in X columns")

        dates = pd.to_datetime(X[self.date_col]).reset_index(drop=True)
        idx = np.arange(len(dates))

        train_end = self.start_date + pd.DateOffset(
            months=self.train_min_months
        )

        for fold in range(self.n_splits):
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Embargo: shift train_mask cutoff back by embargo_days so that
            # rolling features (computed on the full dataset before fold
            # construction) don't leak across the train->val boundary.
            embargo_cutoff = train_end - pd.DateOffset(days=self.embargo_days)
            train_mask = (dates >= self.start_date) & (dates < embargo_cutoff)
            val_mask = (dates >= val_start) & (dates < val_end)
            test_mask = (dates >= test_start) & (dates < test_end)

            train_idx = idx[train_mask.to_numpy()]
            val_idx = idx[val_mask.to_numpy()]
            test_idx = idx[test_mask.to_numpy()]

            if len(test_idx) == 0:
                # Ran out of data
                break

            yield train_idx, val_idx, test_idx

            # Expand training window by advancing train_end
            train_end = test_end

    # For sklearn compatibility (not really used — we yield triples)
    def _iter_test_indices(self, X, y=None, groups=None):
        for _, _, test_idx in self.split(X, y, groups):
            yield test_idx


# ---------- Leave-county-out spatial splitter ----------

class LeaveCountyOutSplit(BaseCrossValidator):
    """Spatial cross-validation: hold out N counties per fold.

    Tests whether the model generalizes to geographic regions it has never
    seen. Expected finding: spatial CV numbers are WORSE than temporal CV
    numbers because geographic heterogeneity is real — that's a publishable
    methodological observation.

    Parameters
    ----------
    county_col : str
        Name of the column holding the county (or agency proxy).
    n_holdout : int
        Number of counties to hold out per fold.
    n_splits : int
        Number of spatial folds.
    random_state : int
        RNG seed for deterministic fold assignment.
    """

    def __init__(
        self,
        county_col: str = "sub_agency",
        n_holdout: int = 3,
        n_splits: int = 8,
        random_state: int = 42,
    ):
        self.county_col = county_col
        self.n_holdout = n_holdout
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterable[
        Tuple[np.ndarray, np.ndarray]
    ]:
        if self.county_col not in X.columns:
            raise ValueError(
                f"{self.county_col} not in X columns. "
                f"Consider using 'sub_agency' or 'agency' as a county proxy."
            )

        rng = np.random.default_rng(self.random_state)
        all_counties = X[self.county_col].dropna().unique().tolist()
        rng.shuffle(all_counties)
        idx = np.arange(len(X))

        # Pick the first (n_splits * n_holdout) counties and rotate
        n_needed = self.n_splits * self.n_holdout
        if len(all_counties) < n_needed:
            print(
                f"WARNING: only {len(all_counties)} counties available, "
                f"{n_needed} needed for {self.n_splits} folds × "
                f"{self.n_holdout} holdout each. Reducing splits."
            )
            self.n_splits = len(all_counties) // self.n_holdout

        for fold in range(self.n_splits):
            start = fold * self.n_holdout
            holdout = set(all_counties[start : start + self.n_holdout])
            test_mask = X[self.county_col].isin(holdout).to_numpy()
            train_mask = ~test_mask
            yield idx[train_mask], idx[test_mask]

    def _iter_test_indices(self, X, y=None, groups=None):
        for _, test_idx in self.split(X, y, groups):
            yield test_idx


# ---------- Combined walk-forward + spatial ----------

class CombinedSplit(BaseCrossValidator):
    """The honest generalization bound: train on past years of N-k counties,
    test on future year of the held-out k counties.

    For every (temporal fold, spatial fold) pair, produces a split where
    training is the intersection (past AND non-holdout counties) and testing
    is the intersection (future AND holdout counties).
    """

    def __init__(
        self,
        temporal: WalkForwardSplit,
        spatial: LeaveCountyOutSplit,
    ):
        self.temporal = temporal
        self.spatial = spatial

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.temporal.n_splits * self.spatial.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None):
        idx = np.arange(len(X))

        for t_fold, (t_train, t_val, t_test) in enumerate(
            self.temporal.split(X)
        ):
            for s_fold, (s_train, s_test) in enumerate(
                self.spatial.split(X)
            ):
                # Train: past AND non-holdout county
                train_set = set(t_train.tolist()) & set(s_train.tolist())
                # Val: past AND non-holdout county (validation stays in-sample spatially)
                val_set = set(t_val.tolist()) & set(s_train.tolist())
                # Test: future AND holdout county
                test_set = set(t_test.tolist()) & set(s_test.tolist())

                train_idx = np.array(sorted(train_set))
                val_idx = np.array(sorted(val_set))
                test_idx = np.array(sorted(test_set))

                if len(test_idx) == 0 or len(train_idx) == 0:
                    continue

                yield train_idx, val_idx, test_idx

    def _iter_test_indices(self, X, y=None, groups=None):
        for _, _, test_idx in self.split(X, y, groups):
            yield test_idx


# ---------- Orchestrator ----------

def materialize_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per original stop + fold assignment
    columns for each strategy.

    Columns:
      - stop_id (matches df.id)
      - temporal_fold_0 .. temporal_fold_{N-1}: 'train' / 'val' / 'test' / NaN
      - spatial_fold_0 .. spatial_fold_{M-1}: 'train' / 'test' / NaN
      - combined_fold_{i}_{j}: 'train' / 'val' / 'test' / NaN
    """
    out = pd.DataFrame({"stop_id": df["id"].values})
    n = len(df)

    # ---- Walk-forward temporal ----
    print("\n[Walk-forward temporal CV]")
    temporal = WalkForwardSplit(
        date_col="stop_date",
        start_date=config.WALK_FORWARD_START,
        train_min_months=config.WALK_FORWARD_TRAIN_MIN_MONTHS,
        val_months=config.WALK_FORWARD_VAL_MONTHS,
        test_months=config.WALK_FORWARD_TEST_MONTHS,
        n_splits=config.WALK_FORWARD_N_SPLITS,
    )
    for fold_i, (tr, va, te) in enumerate(temporal.split(df)):
        col = f"temporal_fold_{fold_i}"
        # Use object dtype so we can freely assign train/val/test/none
        out[col] = pd.Series(["none"] * n, dtype="object")
        out.loc[tr, col] = "train"
        out.loc[va, col] = "val"
        out.loc[te, col] = "test"
        # Cast to category after assignment to save space
        out[col] = out[col].astype(
            pd.CategoricalDtype(categories=["none", "train", "val", "test"])
        )
        print(
            f"  Fold {fold_i}: train={len(tr):>7,}  val={len(va):>6,}  "
            f"test={len(te):>6,}"
        )

    # ---- Spatial (leave-county-out) ----
    print("\n[Spatial leave-county-out CV]")
    try:
        spatial = LeaveCountyOutSplit(
            county_col="sub_agency",
            n_holdout=config.SPATIAL_HOLDOUT_COUNTIES,
            n_splits=config.SPATIAL_N_SPLITS,
            random_state=config.CV_RANDOM_STATE,
        )
        for fold_i, (tr, te) in enumerate(spatial.split(df)):
            col = f"spatial_fold_{fold_i}"
            out[col] = pd.Series(["none"] * n, dtype="object")
            out.loc[tr, col] = "train"
            out.loc[te, col] = "test"
            out[col] = out[col].astype(
                pd.CategoricalDtype(categories=["none", "train", "test"])
            )
            print(
                f"  Fold {fold_i}: train={len(tr):>7,}  test={len(te):>6,}"
            )
    except Exception as e:
        print(f"  Spatial CV skipped: {e}")

    # ---- Combined ----
    # Too many folds for materialization — skip for now unless needed
    # (temporal × spatial = 40 fold columns adds significant width)
    print("\n[Combined CV] — skipped materialization, will compute lazily")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=["temporal", "spatial", "combined", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    in_path = config.PATH_STOPS_FEATURES
    out_path = config.PATH_FOLDS

    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run Stage 03 first.")
        return 2

    if out_path.exists() and not args.force:
        print(f"ERROR: {out_path} already exists. Use --force to overwrite.")
        return 2

    print("=" * 60)
    print("Stage 04 — Build CV folds (walk-forward + spatial)")
    print("=" * 60)

    print(f"  Loading {in_path} ...")
    df = pd.read_parquet(in_path, columns=["id", "stop_date", "sub_agency"])
    print(f"  {len(df):,} stops")

    folds = materialize_folds(df)

    print(f"\n  Writing parquet → {out_path}")
    folds.to_parquet(out_path, compression="zstd", index=False)
    print(f"  Wrote {len(folds):,} rows × {len(folds.columns)} cols")
    print("\nStage 04 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
