"""Multi-task datasets for the unified enforcement prediction model.

Dataset classes:
  - OccurrenceDataset: per-(cell, hour, day) bins → occurrence head (Phase 1)
  - SpeedDataset: per-stop features → speed head (Phase 1)
  - TemporalOccurrenceDataset: per-(cell, date, hour) with 168h lookback → Phase 2

Both share the same cell-level features (road context, POI distances,
enforcement composition). The cell features are the input to the shared
FT-Transformer backbone; the per-stop features are additional input to
the speed head only.

The training loop alternates between occurrence and speed batches,
updating the shared backbone on both signals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Fold-safe target encoder. Runtime TE with smoothing replaces the
# precomputed *_te_fold0 columns that caused the speed-AUC leak (see §3
# of the executive report — Leak A).
from src.data._encoders import FoldSafeTargetEncoder


# ---------- Feature specifications ----------

# Cell-level numeric features (shared backbone input)
# These are the features that describe a LOCATION, not a specific stop.
# When loading from bin parquets (07a outputs), these columns are already
# aggregated. When loading from stops_features per-stop, most have a cell-
# level name mapping in STOP_TO_CELL_MAP (inside SpeedDataset / StopMultiDataset).
CELL_NUMERIC_FEATURES = [
    # Location
    "cell_lat", "cell_lng",
    # Road
    "road_maxspeed_mean", "road_maxspeed_max",
    "road_curvature_mean", "road_sinuosity_mean",
    "road_max_grade_max", "road_lanes_mean",
    "is_highway_cell", "highway_stop_frac",
    # POI distance
    "dist_police_mean", "dist_school_mean", "dist_hospital_mean",
    "dist_fire_mean", "dist_court_mean",
    # Rolling history
    "stops_7d_mean", "stops_30d_mean", "stops_90d_mean",
    # Composition (leaky for per-stop heads — zeroed in StopMultiDataset)
    "speed_frac", "search_frac",
    "radar_frac", "laser_frac", "patrol_frac",
    # Holiday composition
    "us_federal_holiday_frac", "dui_crackdown_holiday_frac",
    "travel_holiday_frac", "holiday_weekend_frac",
    # Speed camera proximity
    "dist_speed_camera_mean", "near_speed_camera_frac",
    "cameras_within_500m_mean",
    # AADT (traffic exposure)
    "aadt_log_max", "aadt_log_mean", "aadt_f_system_min",
    "aadt_truck_pct_max", "aadt_reliable_frac",
    # Crash history
    "crashes_500m_mean", "crashes_1km_mean",
    "fatal_1km_max", "injury_500m_mean", "crash_hotspot_frac",
    # Weather (per-bin means from add_weather_features)
    "temp_c_mean", "precip_mean", "rain_frac", "heavy_rain_frac",
    "snow_frac", "visibility_mean", "wind_speed_mean",
    # Demographic + reg-state cell aggregates (from add_demographic_features)
    "csv_era_stops", "belts_frac", "commercial_veh_frac", "hazmat_frac",
    "contrib_accident_frac", "male_frac", "reg_state_MD_frac",
    "reg_state_far_frac", "dl_state_diversity", "race_entropy",
    # Lens stats — behavioral signals from add_lens_stats_features.
    # v3.3.0: citation_rate is FOLD-AWARE — swapped per fold by
    # deep/train_multi.load_fold_split + deep/predict_multi_ensemble.
    # build_fold_dataset (each fold gets citation_rate_fold{N} written
    # into the canonical citation_rate column). Safe for ALL heads
    # including disposition. strictness_* still derive from is_speed_related
    # filter — leaky for the speed head; revisit in v3.4.0.
    "citation_rate", "strictness_speed_over", "strictness_p75",
    "frac_luxury", "frac_performance", "frac_motorcycle",
    "frac_pickup", "frac_commercial",
    # Phase 0A/E — arrest-type axis aggregates (orthogonal: marked/unmarked
    # ⊥ stationary/mobile). Populated by 07a_*.py detect_agg/comp_agg.
    "stationary_trap_frac", "mobile_enforcement_frac",
    "marked_frac", "unmarked_frac", "automated_frac",
    # Phase 0E — cell-level composite enthusiast-danger features, merged
    # from cell_driver_features.parquet via load_multi_target_bins.
    "canyon_speed_trap_score", "unmarked_radar_frac",
    "laser_extreme_combo_frac", "curvy_night_frac",
    # Phase 0D — vehicle-targeting log-ratios (county-baseline-normalized).
    # Positive = cell over-cites this vehicle type relative to MoCo average.
    "luxury_targeting_logratio", "performance_targeting_logratio",
    "motorcycle_targeting_logratio", "out_of_state_targeting_logratio",
    "enthusiast_targeting_score", "unusual_vehicle_score",
    # Phase 0D — cell_driver_features signals not previously consumed:
    # organized (laser+radar+vascar mix), laser_dominance (laser - max(
    # radar, patrol)), intent_score (organized × excessive), excessive/
    # extreme speed-over fractions.
    "organized_frac", "laser_dominance", "intent_score",
    "excessive_speed_frac", "extreme_speed_frac",
]

# Cell-level categorical features (need embedding lookup)
CELL_CAT_FEATURES: List[str] = []  # currently none at bin level; road_class/surface are pre-aggregated as means

# Time features for the occurrence head
TIME_FEATURES = [
    "hour", "day_of_week",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rush_morning", "is_rush_evening",
    "is_night", "is_late_night", "is_school_hour",
]

# Per-stop numeric features for the speed head (EXCLUDES leakage cols)
STOP_NUMERIC_FEATURES = [
    "hour", "minute", "day_of_week", "month", "quarter", "year",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_rush_hour_morning", "is_rush_hour_evening",
    "is_night", "is_late_night", "is_school_zone_hour", "is_bar_close_hour",
    "is_us_federal_holiday", "days_until_weekend",
    "days_since_dataset_start",
    # `is_covid_era` / `is_post_covid` are kept in the list only to
    # preserve the input dim for already-trained checkpoints (fold 0
    # was trained with them). The std floor guard in StopMultiDataset
    # mutes them to ~0 signal in walk-forward folds (constant in train
    # -> std=0 -> muted to std=1, plus clip to +/-10 on z-score). If
    # we ever retrain all folds from scratch, drop these two.
    "is_covid_era", "is_post_covid",
    "hour_since_shift_start",
    # Road
    "road_maxspeed", "road_lanes", "road_curvature", "road_sinuosity",
    "road_max_turn", "road_switchbacks",
    "road_elev_gain", "road_elev_loss", "road_max_grade",
    "road_clubsport", "road_backroads", "road_grandtour",
    "snap_distance_m", "curvature_per_mi",
    "stop_on_curvy_road", "stop_on_mountain_pass",
    # POI
    "dist_police_m", "dist_school_m", "dist_hospital_m",
    "dist_fire_station_m", "dist_courthouse_m", "dist_university_m",
    "dist_park_m", "dist_mall_m",
    "log_dist_police_m", "log_dist_school_m", "log_dist_hospital_m",
    "is_in_school_zone", "is_near_hospital", "is_near_police_station",
    "is_near_courthouse", "is_in_urban_core",
    "school_zone_during_hours", "urban_core_late_night",
    "avg_dist_top3_poi_m",
    # Rolling
    "stops_last_7d", "stops_last_30d", "stops_last_90d",
    "days_since_last_stop_here", "stops_trending_up",
    # Vehicle
    "vehicle_age", "is_truck", "is_motorcycle", "is_suv", "is_sedan",
    "is_van", "is_commercial", "is_luxury", "is_performance",
    "vehicle_year_decade",
    # Driver
    "is_out_of_state", "is_neighbor_state", "is_far_state",
    # Agency
    "agency_is_traffic_focused",
    # Holiday flags (per-stop booleans)
    "is_dui_crackdown_holiday", "is_travel_holiday", "is_holiday_weekend",
    "days_to_nearest_holiday",
    # Speed camera proximity (per-stop)
    "dist_speed_camera_m", "is_near_speed_camera_500m",
    "speed_cameras_within_500m",
    # AADT (per-stop)
    "aadt_log", "aadt_f_system", "aadt_truck_pct", "aadt_is_reliable",
    # Crash history (per-stop)
    "crashes_within_500m", "crashes_within_1km",
    "fatal_crashes_within_1km", "injury_crashes_within_500m",
    "is_crash_hotspot",
    # Weather (per-stop, from add_weather_features)
    "temperature_c", "precipitation_mm", "is_rain", "is_heavy_rain",
    "is_snow", "visibility_m", "wind_speed_kph",
    # Demographic + reg-state cell aggregates (broadcast to per-stop in parquet)
    "belts_frac", "commercial_veh_frac", "hazmat_frac",
    "contrib_accident_frac", "male_frac", "reg_state_MD_frac",
    "reg_state_far_frac", "dl_state_diversity", "race_entropy",
    "csv_era_stops",
    # Lens stats per-cell (broadcast to per-stop). v3.3.0: citation_rate is
    # FOLD-AWARE — swapped per fold in load_fold_split / build_fold_dataset.
    "citation_rate", "strictness_speed_over", "strictness_p75",
    "frac_luxury", "frac_performance", "frac_motorcycle",
    "frac_pickup", "frac_commercial",
    # NOTE: the 19 precomputed *_te_fold0 columns were removed 2026-04-23.
    # They were computed without Bayesian smoothing and gave near-deterministic
    # lookups for categories like violation_type=Speeding, causing XGB to hit
    # AUC=1.000 on fold 0 speed_auc (speed_leak_localization.py). Replaced at
    # runtime inside SpeedDataset / StopMultiDataset by FoldSafeTargetEncoder
    # (smoothing=20.0), matching Stage 05's honest 0.842 XGB baseline. See
    # docs/speed-auc-leak-investigation-2026-04-23.md.
]

# Leak-safe variant of STOP_NUMERIC_FEATURES. Only drops the era flags
# that cause the walk-forward 0.500-AUC bug if the std-floor guard fails.
# The TE-speed drops that used to be here are now unnecessary because the
# precomputed TE columns were removed from STOP_NUMERIC_FEATURES entirely
# (2026-04-23 refactor — see above).
STOP_NUMERIC_FEATURES_LEAK_SAFE = [
    c for c in STOP_NUMERIC_FEATURES if c not in {
        "is_covid_era", "is_post_covid",
    }
]


# Raw categoricals that get fold-safe target encoded at runtime. Mirrors the
# Stage 05 baseline's TARGET_ENCODE_COLS so the FT-Transformer and XGBoost
# compare on the same feature representation.
TE_CATEGORICALS = [
    "sub_agency",         # ~9 MCP districts
    "vehicle_type",       # ~35 (truck, motorcycle, SUV, ...)
    "vehicle_make",       # ~3760
    "road_class",         # ~15 (motorway, primary, residential, ...)
    "road_surface",       # ~10 (asphalt, concrete, dirt, ...)
    "arrest_type_letter", # 19 letters A-S + UNK (v3.3.0 — replaces
                          # violation_type/_top which were 4-way Citation/
                          # Warning/SERO/ESERO. Those LEAKED is_citation
                          # because each violation_type value got a unique
                          # 4-tuple of TE values across the 4 binary heads,
                          # letting the disposition head perfectly recover
                          # is_citation = (violation_type == "Citation").
                          # arrest_type_letter has only 0.32-0.92 citation
                          # rate spread across letters — no perfect lookup.)
]

# Binary targets to fit runtime TE against. Must match STOP_BINARY_TARGETS
# below (one encoder per target means 7 cats * 4 targets = 28 runtime columns).
#
# DO NOT ADD `is_citation` HERE. TE_CATEGORICALS includes `violation_type` and
# `violation_type_top` — adding `is_citation` to this list would create
# `violation_type_te_is_citation` which is direct target leakage (is_citation
# is derived from violation_type=="Citation"). v3.3.0 disposition head learns
# from arrest_type_letter + fold-aware citation_rate (see deep/train_multi.
# load_fold_split + 09j_precompute_cell_lens_stats.py --fold-idx).
_TE_BINARY_TARGETS = [
    "is_speed_related", "search_conducted", "accident", "personal_injury",
]


def _fit_te_encoders(stops_df: pd.DataFrame, targets: list) -> Dict[str, FoldSafeTargetEncoder]:
    """Fit one FoldSafeTargetEncoder per target on the given train slice.

    Returned dict is reused at val/test time via `stats["te_encoders"]` so the
    per-fold TE values are computed on TRAIN ONLY — no leakage.
    """
    present_cats = [c for c in TE_CATEGORICALS if c in stops_df.columns]
    out: Dict[str, FoldSafeTargetEncoder] = {}
    for tgt in targets:
        if tgt not in stops_df.columns:
            continue
        enc = FoldSafeTargetEncoder(cols=present_cats, smoothing=20.0)
        enc.fit(stops_df, stops_df[tgt].astype(float).fillna(0))
        out[tgt] = enc
    return out


def _augment_with_te(
    stops_df: pd.DataFrame,
    te_encoders: Dict[str, FoldSafeTargetEncoder],
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply fitted TE encoders to stops_df, return (augmented_df, new_col_names).

    Output columns are named `{cat}_te_{target}` so they're unique across targets.
    """
    aug = stops_df.copy()
    new_cols: List[str] = []
    for tgt, enc in te_encoders.items():
        present_cats = [c for c in enc.cols if c in aug.columns]
        if not present_cats:
            continue
        te_df = enc.transform(aug[present_cats])
        # FoldSafeTargetEncoder.transform drops the original cat cols and adds
        # `{col}_te` (single target). We rename per target so multiple targets
        # can coexist without colliding.
        for cat in present_cats:
            src = f"{cat}_te"
            dst = f"{cat}_te_{tgt}"
            if src in te_df.columns:
                aug[dst] = te_df[src].values.astype("float32")
                new_cols.append(dst)
    return aug, new_cols


# Per-stop categorical features for the speed head.
# Cardinalities sized with headroom (~2x current max per column in the
# 1.24M-row stops_features parquet — April 2026) so later folds can
# encode newly-seen values without embedding OOB. Actual train-time
# cardinalities are always <= these caps; unused indices are dead
# embeddings (no training signal, small memory cost).
STOP_CAT_FEATURES = [
    "sub_agency",          # full: 9    cap: 16
    "vehicle_type",        # full: 35   cap: 64
    "vehicle_make",        # full: 3760 cap: 4096
    "road_class",          # full: 15   cap: 32
    "road_surface",        # full: 10   cap: 16
    "arrest_type_letter",  # full: 19+UNK=20   cap: 32   (v3.3.0 — added; replaces
                           #     violation_type/violation_type_top which were dropped
                           #     from inputs because violation_type is now a TARGET via
                           #     the disposition head — would be target leakage to keep
                           #     it as input. Letters observed in 1.27M-row data:
                           #     A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S + UNK fallback.)
]

STOP_CAT_CARDINALITIES = [16, 64, 4096, 32, 16, 32]


# ---------- Datasets ----------

def load_multi_target_bins(
    occurrence_path: Path,
    speed_path: Path,
    trap_path: Path,
) -> pd.DataFrame:
    """Merge the 3 bin parquets produced by 07a_* into one with 3 count cols.

    The join key is (h3_cell, hour, day_of_week). Speed/trap counts are
    filled with 0 where a bin exists in the total-occurrence table but not
    in the narrower speed/trap tables (i.e., cells where speed enforcement
    hasn't happened get speed_count=0 which is correct).
    """
    occ = pd.read_parquet(occurrence_path)
    occ = occ.rename(columns={"stop_count": "stop_count_total"})
    spd = pd.read_parquet(speed_path, columns=["h3_cell", "hour", "day_of_week", "stop_count"])
    spd = spd.rename(columns={"stop_count": "stop_count_speed"})
    trap = pd.read_parquet(trap_path, columns=["h3_cell", "hour", "day_of_week", "stop_count"])
    trap = trap.rename(columns={"stop_count": "stop_count_trap"})

    merged = occ.merge(spd, on=["h3_cell", "hour", "day_of_week"], how="left")
    merged = merged.merge(trap, on=["h3_cell", "hour", "day_of_week"], how="left")
    merged["stop_count_speed"] = merged["stop_count_speed"].fillna(0)
    merged["stop_count_trap"] = merged["stop_count_trap"].fillna(0)

    # Phase 0F — merge cell-level features from cell_driver_features.parquet
    # (canyon_speed_trap_score, log-ratios, laser_dominance, intent_score,
    # etc.) onto the bin-level frame. These are constant per h3_cell (do
    # not vary by hour/dow) but they need to be present for the FT-T to
    # consume them as CELL_NUMERIC_FEATURES. Missing cells get 0.
    cell_feat_path = occurrence_path.parent / "cell_driver_features.parquet"
    if cell_feat_path.exists():
        _cell_cols = [
            "h3_cell",
            # Phase 0E cell composites
            "canyon_speed_trap_score", "unmarked_radar_frac",
            "laser_extreme_combo_frac", "curvy_night_frac",
            # Phase 0D vehicle-targeting
            "luxury_targeting_logratio", "performance_targeting_logratio",
            "motorcycle_targeting_logratio", "out_of_state_targeting_logratio",
            "enthusiast_targeting_score", "unusual_vehicle_score",
            # Previously unused cell_driver_features signals
            "organized_frac", "laser_dominance", "intent_score",
            "excessive_speed_frac", "extreme_speed_frac",
        ]
        cdf = pd.read_parquet(cell_feat_path)
        available = [c for c in _cell_cols if c in cdf.columns]
        cdf = cdf[available]
        merged = merged.merge(cdf, on="h3_cell", how="left")
        # Fill for any cell in bins but missing from cell_driver_features
        # (ultra-sparse cells that 09g's n_stops < 3 filter dropped).
        for c in available:
            if c != "h3_cell":
                merged[c] = merged[c].fillna(0.0)
    return merged


class OccurrenceDataset(Dataset):
    """Dataset for the 3 Poisson heads: per-(cell, hour, day) bins.

    Emits 3 targets per bin:
        target_total  — stop_count over all stops
        target_speed  — stop_count among is_speed_related=1
        target_trap   — stop_count among laser+radar detections

    If the input df has only `stop_count` (legacy Phase 1), speed/trap
    default to 0 and the loss on those heads becomes degenerate — train.py
    should skip those heads when that's the case.
    """

    def __init__(self, bins_df: pd.DataFrame, stats: Optional[Dict] = None):
        # Tolerate missing new feature cols (e.g. when bins from an older
        # 07a run don't have weather/demographic aggregates yet)
        available = [c for c in CELL_NUMERIC_FEATURES if c in bins_df.columns]
        missing = set(CELL_NUMERIC_FEATURES) - set(available)
        if missing:
            print(f"  [OccurrenceDataset] {len(missing)} cell features "
                  f"missing from bins — filling zeros: {sorted(missing)[:5]}...")
        for c in missing:
            bins_df = bins_df.copy()
            bins_df[c] = 0.0
        self.feature_order = CELL_NUMERIC_FEATURES

        raw_cell = bins_df[self.feature_order].fillna(0).astype("float32").values
        raw_time = bins_df[TIME_FEATURES].fillna(0).astype("float32").values

        # Standardize numeric features (zero mean, unit variance).
        # Use std floor 1.0 for constant-in-train features (mute them)
        # and clip normalized values to +/-10 — see StopMultiDataset
        # for the full explanation.
        if stats is None:
            self.cell_mean = raw_cell.mean(axis=0)
            cell_std_raw = raw_cell.std(axis=0)
            self.cell_std = np.where(cell_std_raw < 1e-3, 1.0, cell_std_raw)
            self.time_mean = raw_time.mean(axis=0)
            time_std_raw = raw_time.std(axis=0)
            self.time_std = np.where(time_std_raw < 1e-3, 1.0, time_std_raw)
        else:
            self.cell_mean = stats["cell_mean"]
            self.cell_std = stats["cell_std"]
            self.time_mean = stats["time_mean"]
            self.time_std = stats["time_std"]

        cell_norm = (raw_cell - self.cell_mean) / self.cell_std
        np.clip(cell_norm, -10.0, 10.0, out=cell_norm)
        self.cell_numeric = torch.tensor(cell_norm, dtype=torch.float32)
        time_norm = (raw_time - self.time_mean) / self.time_std
        np.clip(time_norm, -10.0, 10.0, out=time_norm)
        self.time_features = torch.tensor(time_norm, dtype=torch.float32)

        # Targets — support both the old "stop_count" schema (Phase 1)
        # and the new 3-target schema (Phase 2 via load_multi_target_bins)
        total_col = ("stop_count_total" if "stop_count_total" in bins_df.columns
                     else "stop_count")
        self.target_total = torch.tensor(
            bins_df[total_col].astype("float32").values, dtype=torch.float32,
        )
        self.target_speed = torch.tensor(
            bins_df.get("stop_count_speed", pd.Series(0.0, index=bins_df.index))
                   .astype("float32").values,
            dtype=torch.float32,
        )
        self.target_trap = torch.tensor(
            bins_df.get("stop_count_trap", pd.Series(0.0, index=bins_df.index))
                   .astype("float32").values,
            dtype=torch.float32,
        )

    def get_stats(self) -> Dict:
        return {
            "cell_mean": self.cell_mean, "cell_std": self.cell_std,
            "time_mean": self.time_mean, "time_std": self.time_std,
        }

    def __len__(self) -> int:
        return len(self.target_total)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "cell_numeric": self.cell_numeric[idx],
            "cell_categorical": torch.zeros(0, dtype=torch.long),  # no cell cats
            "time_features": self.time_features[idx],
            # "target" is an alias of target_total for train.py compatibility
            # (train_epoch was written for TemporalOccurrenceDataset which returns
            # a single 'target' key; this keeps Phase 1 training working while
            # preserving the 3 separate Poisson targets for downstream use).
            "target": self.target_total[idx],
            "target_total": self.target_total[idx],
            "target_speed": self.target_speed[idx],
            "target_trap": self.target_trap[idx],
            "task": torch.tensor(0),  # 0 = occurrence (3 Poisson heads)
        }


class SpeedDataset(Dataset):
    """Dataset for the speed head: per-stop binary classification."""

    def __init__(self, stops_df: pd.DataFrame, cat_encoders: Optional[Dict] = None, stats: Optional[Dict] = None):
        # Build category → integer mappings if not provided
        if cat_encoders is None:
            self.cat_encoders = {}
            for col in STOP_CAT_FEATURES:
                if col in stops_df.columns:
                    uniques = stops_df[col].astype(str).unique()
                    self.cat_encoders[col] = {v: i for i, v in enumerate(uniques)}
        else:
            self.cat_encoders = cat_encoders

        # Runtime fold-safe TE encoder for is_speed_related only (Phase 1 path).
        # Mirrors StopMultiDataset's TE integration; see `_fit_te_encoders`.
        if stats is not None and "te_encoders" in stats:
            self.te_encoders = stats["te_encoders"]
        else:
            self.te_encoders = _fit_te_encoders(stops_df, ["is_speed_related"])
        stops_df_aug, te_col_names = _augment_with_te(stops_df, self.te_encoders)
        self._te_col_names = te_col_names

        # Numeric features — standardized
        effective_numeric = STOP_NUMERIC_FEATURES + te_col_names
        available_numeric = [c for c in effective_numeric if c in stops_df_aug.columns]
        raw_stop = stops_df_aug[available_numeric].fillna(0).astype("float32").values
        # Mirror the normalization gotcha guards from StopMultiDataset / OccurrenceDataset:
        # raise std floor to 1.0 for near-constant features (std < 1e-3) and clip
        # normalized values to [-10, 10]. Prevents the 1M× z-score amplification
        # that can freeze binary AUCs at 0.500 in walk-forward folds.
        if stats is None:
            self.stop_mean = raw_stop.mean(axis=0)
            train_std_raw = raw_stop.std(axis=0)
            self.stop_std = np.where(train_std_raw < 1e-3, 1.0, train_std_raw)
        else:
            self.stop_mean = stats.get("stop_mean", raw_stop.mean(axis=0))
            self.stop_std = stats.get(
                "stop_std",
                np.where(raw_stop.std(axis=0) < 1e-3, 1.0, raw_stop.std(axis=0)),
            )
        normalized = (raw_stop - self.stop_mean) / self.stop_std
        np.clip(normalized, -10.0, 10.0, out=normalized)
        self.stop_numeric = torch.tensor(normalized, dtype=torch.float32)
        self.n_stop_numeric = len(available_numeric)

        # Categorical features → integer indices
        cat_indices = []
        for col in STOP_CAT_FEATURES:
            if col in stops_df.columns:
                mapping = self.cat_encoders.get(col, {})
                indices = stops_df[col].astype(str).map(mapping).fillna(0).astype(int).values
                cat_indices.append(indices)
            else:
                cat_indices.append(np.zeros(len(stops_df), dtype=int))
        self.stop_categorical = torch.tensor(
            np.column_stack(cat_indices), dtype=torch.long
        )

        # Cell-level features — must match CELL_NUMERIC_FEATURES exactly so the
        # shared backbone gets the same dimensionality from both datasets.
        # Map per-stop column names to cell-level equivalents.
        STOP_TO_CELL_MAP = {
            "cell_lat": "latitude", "cell_lng": "longitude",
            "road_maxspeed_mean": "road_maxspeed", "road_maxspeed_max": "road_maxspeed",
            "road_curvature_mean": "road_curvature", "road_sinuosity_mean": "road_sinuosity",
            "road_max_grade_max": "road_max_grade", "road_lanes_mean": "road_lanes",
            "dist_police_mean": "dist_police_m", "dist_school_mean": "dist_school_m",
            "dist_hospital_mean": "dist_hospital_m",
            "dist_fire_mean": "dist_fire_station_m", "dist_court_mean": "dist_courthouse_m",
            "stops_7d_mean": "stops_last_7d", "stops_30d_mean": "stops_last_30d",
            "stops_90d_mean": "stops_last_90d",
            # INTENTIONALLY ZEROED: composition features leak the target
            # for per-stop speed classification. A cell with speed_frac=0.9
            # trivially predicts "speed" for every stop. These features are
            # valid for the occurrence head (cell-level prediction) but are
            # leakage for per-stop classification.
            # "speed_frac": ZEROED
            # "search_frac": ZEROED
            # "radar_frac": ZEROED
            # "laser_frac": ZEROED
            # "patrol_frac": ZEROED
            # "highway_stop_frac": ZEROED
        }
        cell_data = np.zeros((len(stops_df), len(CELL_NUMERIC_FEATURES)), dtype=np.float32)
        for i, cell_col in enumerate(CELL_NUMERIC_FEATURES):
            stop_col = STOP_TO_CELL_MAP.get(cell_col, cell_col)
            if stop_col in stops_df.columns:
                cell_data[:, i] = stops_df[stop_col].fillna(0).astype("float32").values
        # Standardize cell numeric — same gotcha guards as stop features above.
        if stats is None:
            self.cell_mean = cell_data.mean(axis=0)
            cell_std_raw = cell_data.std(axis=0)
            self.cell_std = np.where(cell_std_raw < 1e-3, 1.0, cell_std_raw)
        else:
            self.cell_mean = stats.get("cell_mean_spd", cell_data.mean(axis=0))
            self.cell_std = stats.get(
                "cell_std_spd",
                np.where(cell_data.std(axis=0) < 1e-3, 1.0, cell_data.std(axis=0)),
            )
        cell_norm = (cell_data - self.cell_mean) / self.cell_std
        np.clip(cell_norm, -10.0, 10.0, out=cell_norm)
        self.cell_numeric = torch.tensor(cell_norm, dtype=torch.float32)
        self.n_cell_numeric = len(CELL_NUMERIC_FEATURES)

        # Time features for backbone context
        time_available = [c for c in TIME_FEATURES if c in stops_df.columns]
        self.time_features = torch.tensor(
            stops_df[time_available].fillna(0).astype("float32").values,
            dtype=torch.float32,
        )

        # Target
        self.targets = torch.tensor(
            stops_df["is_speed_related"].astype("float32").values,
            dtype=torch.float32,
        )

    def get_stats(self) -> Dict:
        return {
            "stop_mean": self.stop_mean, "stop_std": self.stop_std,
            "cell_mean_spd": self.cell_mean, "cell_std_spd": self.cell_std,
            "te_encoders": self.te_encoders,
        }

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "cell_numeric": self.cell_numeric[idx],
            "cell_categorical": torch.zeros(0, dtype=torch.long),
            "time_features": self.time_features[idx],
            "stop_numeric": self.stop_numeric[idx],
            "stop_categorical": self.stop_categorical[idx],
            "target": self.targets[idx],
            "task": torch.tensor(1),  # 1 = speed
        }


# ---------- Phase 2: StopMulti for 4 binary heads ----------

# All 4 per-stop binary targets the FT-Transformer emits heads for.
STOP_BINARY_TARGETS = [
    "is_speed_related",
    "search_conducted",
    "accident",
    "personal_injury",
    "is_citation",  # v3.3.0 — disposition head: 1 if Citation else 0 (Warning/SERO/ESERO)
]


class StopMultiDataset(Dataset):
    """Dataset for the 4 per-stop binary heads.

    Emits all 4 binary targets per stop + the same cell/stop/time features
    as SpeedDataset. All composition features (speed_frac/search_frac/etc.)
    are zeroed to prevent target leakage — the target values are what those
    composition frac features are measuring.
    """

    def __init__(
        self,
        stops_df: pd.DataFrame,
        cat_encoders: Optional[Dict] = None,
        stats: Optional[Dict] = None,
        leak_safe: bool = False,
    ):
        # Which feature list drives the numeric tensor. `leak_safe=True`
        # swaps in STOP_NUMERIC_FEATURES_LEAK_SAFE which drops the
        # is_speed_related target-encoded columns + the is_covid era flags.
        # The resulting stop_numeric dim is smaller, so checkpoints from
        # leak-safe and non-leak-safe training are NOT interchangeable.
        self._numeric_feature_list = (
            STOP_NUMERIC_FEATURES_LEAK_SAFE if leak_safe else STOP_NUMERIC_FEATURES
        )
        self.leak_safe = leak_safe

        # Category encoders (reused across folds for stable embedding indices)
        if cat_encoders is None:
            self.cat_encoders = {}
            for col in STOP_CAT_FEATURES:
                if col in stops_df.columns:
                    uniques = stops_df[col].astype(str).unique()
                    self.cat_encoders[col] = {v: i for i, v in enumerate(uniques)}
        else:
            self.cat_encoders = cat_encoders

        # Fold-safe runtime TE encoders (fit on train slice, reused on val/test).
        # 7 cats x 4 binary targets = up to 28 new numeric columns, appended
        # to the numeric feature tensor below. Replaces the 19 precomputed
        # *_te_fold0 columns that were dropped from STOP_NUMERIC_FEATURES.
        if stats is not None and "te_encoders" in stats:
            self.te_encoders = stats["te_encoders"]
        else:
            self.te_encoders = _fit_te_encoders(stops_df, _TE_BINARY_TARGETS)
        stops_df_aug, te_col_names = _augment_with_te(stops_df, self.te_encoders)
        self._te_col_names = te_col_names

        # Per-stop numeric features.
        # Normalization gotcha: a feature constant in train (the classic
        # example was `is_post_covid`, now removed — all 0 for pre-2020
        # folds, all 1 in post-2020 test) gets std clipped to ~0; any
        # nonzero value in test produces a z-score of ~1e6, saturating the
        # sigmoid and freezing binary AUCs at 0.500. We defend against this
        # by (a) raising the std floor to 1.0 for constant-in-train features
        # (effectively muting them — they carry no training signal anyway)
        # and (b) clipping the normalized tensor to +/-10 as a safety net
        # for any remaining pathological outlier.
        effective_numeric = self._numeric_feature_list + te_col_names
        available_numeric = [c for c in effective_numeric if c in stops_df_aug.columns]
        raw_stop = stops_df_aug[available_numeric].fillna(0).astype("float32").values
        if stats is None:
            self.stop_mean = raw_stop.mean(axis=0)
            train_std_raw = raw_stop.std(axis=0)
            # If train std < 1e-3, feature is effectively constant; mute it
            # (std=1.0 means z-score = test_value - mean, unamplified).
            self.stop_std = np.where(train_std_raw < 1e-3, 1.0, train_std_raw)
        else:
            self.stop_mean = stats.get("stop_mean", raw_stop.mean(axis=0))
            self.stop_std = stats.get("stop_std",
                np.where(raw_stop.std(axis=0) < 1e-3, 1.0, raw_stop.std(axis=0)))
        normalized = (raw_stop - self.stop_mean) / self.stop_std
        np.clip(normalized, -10.0, 10.0, out=normalized)
        self.stop_numeric = torch.tensor(normalized, dtype=torch.float32)
        self.n_stop_numeric = len(available_numeric)
        self.stop_numeric_features_used = available_numeric

        # Per-stop categoricals → int indices
        cat_indices = []
        for col in STOP_CAT_FEATURES:
            if col in stops_df.columns:
                mapping = self.cat_encoders.get(col, {})
                indices = stops_df[col].astype(str).map(mapping).fillna(0).astype(int).values
                cat_indices.append(indices)
            else:
                cat_indices.append(np.zeros(len(stops_df), dtype=int))
        self.stop_categorical = torch.tensor(
            np.column_stack(cat_indices), dtype=torch.long,
        )

        # Cell-level features — mapped from per-stop columns, composition zeroed
        LEAKY_CELL_FEATURES = {
            "speed_frac", "search_frac",
            "radar_frac", "laser_frac", "patrol_frac",
        }
        STOP_TO_CELL_MAP = {
            "cell_lat": "latitude", "cell_lng": "longitude",
            "road_maxspeed_mean": "road_maxspeed", "road_maxspeed_max": "road_maxspeed",
            "road_curvature_mean": "road_curvature", "road_sinuosity_mean": "road_sinuosity",
            "road_max_grade_max": "road_max_grade", "road_lanes_mean": "road_lanes",
            "dist_police_mean": "dist_police_m", "dist_school_mean": "dist_school_m",
            "dist_hospital_mean": "dist_hospital_m",
            "dist_fire_mean": "dist_fire_station_m", "dist_court_mean": "dist_courthouse_m",
            "stops_7d_mean": "stops_last_7d", "stops_30d_mean": "stops_last_30d",
            "stops_90d_mean": "stops_last_90d",
            "dist_speed_camera_mean": "dist_speed_camera_m",
            "near_speed_camera_frac": "is_near_speed_camera_500m",
            "cameras_within_500m_mean": "speed_cameras_within_500m",
            "aadt_log_max": "aadt_log", "aadt_log_mean": "aadt_log",
            "aadt_f_system_min": "aadt_f_system",
            "aadt_truck_pct_max": "aadt_truck_pct",
            "aadt_reliable_frac": "aadt_is_reliable",
            "crashes_500m_mean": "crashes_within_500m",
            "crashes_1km_mean": "crashes_within_1km",
            "fatal_1km_max": "fatal_crashes_within_1km",
            "injury_500m_mean": "injury_crashes_within_500m",
            "crash_hotspot_frac": "is_crash_hotspot",
            "us_federal_holiday_frac": "is_us_federal_holiday",
            "dui_crackdown_holiday_frac": "is_dui_crackdown_holiday",
            "travel_holiday_frac": "is_travel_holiday",
            "holiday_weekend_frac": "is_holiday_weekend",
            "temp_c_mean": "temperature_c",
            "precip_mean": "precipitation_mm",
            "rain_frac": "is_rain",
            "heavy_rain_frac": "is_heavy_rain",
            "snow_frac": "is_snow",
            "visibility_mean": "visibility_m",
            "wind_speed_mean": "wind_speed_kph",
            # demographic + is_highway_cell + highway_stop_frac stay as-is
            # (they're already cell-level values broadcast per-stop)
        }
        cell_data = np.zeros(
            (len(stops_df), len(CELL_NUMERIC_FEATURES)), dtype=np.float32,
        )
        for i, cell_col in enumerate(CELL_NUMERIC_FEATURES):
            if cell_col in LEAKY_CELL_FEATURES:
                continue  # keep zeros
            stop_col = STOP_TO_CELL_MAP.get(cell_col, cell_col)
            if stop_col in stops_df.columns:
                cell_data[:, i] = stops_df[stop_col].fillna(0).astype("float32").values
        # Same normalization guard as stop_numeric: mute constant-in-train
        # features (std floor 1.0) and clip the normalized tensor to +/-10.
        if stats is None:
            self.cell_mean = cell_data.mean(axis=0)
            cell_std_raw = cell_data.std(axis=0)
            self.cell_std = np.where(cell_std_raw < 1e-3, 1.0, cell_std_raw)
        else:
            self.cell_mean = stats.get("cell_mean_multi", cell_data.mean(axis=0))
            cell_std_default = np.where(
                cell_data.std(axis=0) < 1e-3, 1.0, cell_data.std(axis=0),
            )
            self.cell_std = stats.get("cell_std_multi", cell_std_default)
        cell_norm = (cell_data - self.cell_mean) / self.cell_std
        np.clip(cell_norm, -10.0, 10.0, out=cell_norm)
        self.cell_numeric = torch.tensor(cell_norm, dtype=torch.float32)
        self.n_cell_numeric = len(CELL_NUMERIC_FEATURES)

        # Time features for backbone context
        time_available = [c for c in TIME_FEATURES if c in stops_df.columns]
        self.time_features = torch.tensor(
            stops_df[time_available].fillna(0).astype("float32").values,
            dtype=torch.float32,
        )

        # 4 binary targets — all as float32 for BCEWithLogitsLoss
        targets_raw = {}
        for t in STOP_BINARY_TARGETS:
            if t in stops_df.columns:
                targets_raw[t] = stops_df[t].astype("float32").fillna(0).values
            else:
                print(f"  [StopMultiDataset] target {t} missing — filling 0")
                targets_raw[t] = np.zeros(len(stops_df), dtype="float32")
        self.targets = torch.tensor(
            np.column_stack([targets_raw[t] for t in STOP_BINARY_TARGETS]),
            dtype=torch.float32,
        )
        self.target_names = STOP_BINARY_TARGETS

    def get_stats(self) -> Dict:
        return {
            "stop_mean": self.stop_mean, "stop_std": self.stop_std,
            "cell_mean_multi": self.cell_mean, "cell_std_multi": self.cell_std,
            "te_encoders": self.te_encoders,
        }

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "cell_numeric": self.cell_numeric[idx],
            "cell_categorical": torch.zeros(0, dtype=torch.long),
            "time_features": self.time_features[idx],
            "stop_numeric": self.stop_numeric[idx],
            "stop_categorical": self.stop_categorical[idx],
            "targets": self.targets[idx],  # shape (4,)
            "task": torch.tensor(2),  # 2 = stop_multi (4 binary heads)
        }


# ---------- Phase 2: Temporal Occurrence Dataset ----------

# LSTM input features per timestep
SEQ_FEATURES = 6  # (stop_count, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend)
SEQ_LENGTH = 168  # 1 week of hourly data


class TemporalOccurrenceDataset(Dataset):
    """Occurrence dataset with 168-hour lookback sequences for the LSTM.

    Unlike OccurrenceDataset (weekly templates), this uses actual date-specific
    counts: each sample is (cell, date, hour) → actual_stop_count with a
    168-hour history window.

    Memory: builds per-cell dense hourly arrays. For 1,356 cells × 54,864 hours,
    this is ~280MB float32 — fits easily in RAM.
    """

    def __init__(
        self,
        counts_path: Path,
        meta_path: Path,
        bins_df: pd.DataFrame,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        stats: Optional[Dict] = None,
        nonzero_oversample: float = 5.0,
    ):
        meta = json.loads(meta_path.read_text())
        self.min_date = pd.Timestamp(meta["min_date"])
        self.total_hours = meta["total_hours"]
        self.cells = meta["cells"]
        self.cell_to_idx = {c: i for i, c in enumerate(self.cells)}
        n_cells = len(self.cells)

        # Build dense count arrays: (n_cells, total_hours)
        counts = pd.read_parquet(counts_path)
        self.cell_counts = np.zeros((n_cells, self.total_hours), dtype=np.float32)
        for _, row in counts.iterrows():
            cidx = self.cell_to_idx.get(row["h3_cell"])
            if cidx is None:
                continue
            days_off = (pd.Timestamp(row["date"]) - self.min_date).days
            hidx = days_off * 24 + int(row["hour"])
            if 0 <= hidx < self.total_hours:
                self.cell_counts[cidx, hidx] = row["stop_count"]

        # Build per-cell feature matrix from bins (average over all time slots)
        cell_features_map = {}
        for cell in self.cells:
            cell_bins = bins_df[bins_df["h3_cell"] == cell] if "h3_cell" in bins_df.columns else None
            if cell_bins is not None and len(cell_bins) > 0:
                feats = cell_bins[CELL_NUMERIC_FEATURES].fillna(0).mean().values.astype("float32")
            else:
                feats = np.zeros(len(CELL_NUMERIC_FEATURES), dtype=np.float32)
            cell_features_map[cell] = feats
        raw_cell = np.array([cell_features_map[c] for c in self.cells], dtype=np.float32)

        # Standardize — same gotcha guards as the other datasets.
        if stats is None:
            self.cell_mean = raw_cell.mean(axis=0)
            cell_std_raw = raw_cell.std(axis=0)
            self.cell_std = np.where(cell_std_raw < 1e-3, 1.0, cell_std_raw)
        else:
            self.cell_mean = stats["cell_mean"]
            self.cell_std = stats["cell_std"]
        self.cell_numeric = (raw_cell - self.cell_mean) / self.cell_std  # (n_cells, n_feats)
        np.clip(self.cell_numeric, -10.0, 10.0, out=self.cell_numeric)

        # Build sample index: (cell_idx, hour_idx) pairs within date range
        start_hidx = SEQ_LENGTH  # need 168 hours of history
        end_hidx = self.total_hours

        if date_start:
            ds = (pd.Timestamp(date_start) - self.min_date).days * 24
            start_hidx = max(start_hidx, ds)
        if date_end:
            de = (pd.Timestamp(date_end) - self.min_date).days * 24 + 24
            end_hidx = min(end_hidx, de)

        # Vectorized: find all non-zero (cell, hour) pairs in range
        rng = np.random.default_rng(42)
        valid_slice = self.cell_counts[:, start_hidx:end_hidx]
        nz_cell, nz_hour = np.nonzero(valid_slice > 0)
        nz_hour += start_hidx  # offset back to global indices

        # Non-zero samples (oversampled)
        nz_samples = np.column_stack([nz_cell, nz_hour])
        n_rep = max(1, int(nonzero_oversample))
        nz_oversampled = np.tile(nz_samples, (n_rep, 1))

        # Random zero samples (match non-zero count for balance)
        n_zero = len(nz_oversampled)
        zero_cells = rng.integers(0, n_cells, size=n_zero)
        zero_hours = rng.integers(start_hidx, end_hidx, size=n_zero)
        zero_samples = np.column_stack([zero_cells, zero_hours])

        all_samples = np.concatenate([nz_oversampled, zero_samples], axis=0)
        rng.shuffle(all_samples)

        # Cap at 500K to keep memory/time reasonable
        max_samples = min(len(all_samples), 500_000)
        self.samples = all_samples[:max_samples]

        # Pre-compute day-of-week for the full date range (vectorized)
        self._dow_cache = np.array([
            (self.min_date + pd.Timedelta(days=d)).dayofweek
            for d in range(self.total_hours // 24 + 1)
        ], dtype=np.int32)

    def get_stats(self) -> Dict:
        return {"cell_mean": self.cell_mean, "cell_std": self.cell_std}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cidx, hidx = int(self.samples[idx, 0]), int(self.samples[idx, 1])

        # Cell features (static, pre-standardized)
        cell_num = torch.tensor(self.cell_numeric[cidx], dtype=torch.float32)

        # Time features for the target hour (vectorized)
        h = hidx % 24
        d = self._dow_cache[hidx // 24]
        wknd = d >= 5
        time_feats = np.array([
            h, d,
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
            float(wknd),
            float(not wknd and 6 <= h <= 9),
            float(not wknd and 16 <= h <= 19),
            float(h >= 22 or h <= 5),
            float(0 <= h <= 4),
            float(not wknd and (7 <= h <= 9 or 14 <= h <= 16)),
        ], dtype=np.float32)

        # 168-hour lookback sequence for LSTM (vectorized)
        seq_start = hidx - SEQ_LENGTH
        raw_counts = self.cell_counts[cidx, seq_start:hidx]  # (168,)
        hours = np.arange(seq_start, hidx) % 24
        days = self._dow_cache[np.arange(seq_start, hidx) // 24]

        seq = np.stack([
            raw_counts,
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
            np.sin(2 * np.pi * days / 7),
            np.cos(2 * np.pi * days / 7),
            (days >= 5).astype(np.float32),
        ], axis=1)  # (168, 6)

        target = self.cell_counts[cidx, hidx]

        return {
            "cell_numeric": cell_num,
            "cell_categorical": torch.zeros(0, dtype=torch.long),
            "time_features": torch.tensor(time_feats, dtype=torch.float32),
            "cell_sequence": torch.tensor(seq, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "task": torch.tensor(0),  # occurrence
        }


# ---------- Data loading helpers ----------

def load_occurrence_data(
    bins_path: Path,
    fold_mask: Optional[np.ndarray] = None,
) -> Tuple[OccurrenceDataset, Optional[OccurrenceDataset]]:
    """Load occurrence bins, optionally split by fold mask."""
    bins = pd.read_parquet(bins_path)

    # Ensure required columns exist, fill missing with 0
    for col in CELL_NUMERIC_FEATURES + TIME_FEATURES:
        if col not in bins.columns:
            bins[col] = 0.0

    if fold_mask is not None:
        train_ds = OccurrenceDataset(bins[fold_mask])
        val_ds = OccurrenceDataset(bins[~fold_mask], stats=train_ds.get_stats())
        return train_ds, val_ds
    return OccurrenceDataset(bins), None


def load_speed_data(
    stops_path: Path,
    fold_col: str = "temporal_fold_0",
    folds_path: Optional[Path] = None,
) -> Tuple[SpeedDataset, Optional[SpeedDataset]]:
    """Load per-stop data for speed classification."""
    stops = pd.read_parquet(stops_path)

    # Filter police station stops
    if "dist_police_m" in stops.columns:
        stops = stops[stops["dist_police_m"] >= 100].reset_index(drop=True)

    # Merge folds if provided
    if folds_path is not None:
        folds = pd.read_parquet(folds_path)
        stops = stops.merge(folds, left_on="id", right_on="stop_id", how="inner")

    if fold_col in stops.columns:
        train_mask = stops[fold_col].isin(["train", "val"])
        test_mask = stops[fold_col] == "test"
        # Fold-safe TE fit: fit encoders on TRAIN ONLY (val is also our
        # early-stop signal) — see docs/walk-forward-cv-audit-2026-04-24.md.
        train_only_mask = stops[fold_col] == "train"
        te_encoders = _fit_te_encoders(
            stops[train_only_mask].reset_index(drop=True), ["is_speed_related"]
        )
        # Construct train dataset on train+val with TE encoders from train-only.
        train_ds = SpeedDataset(
            stops[train_mask].reset_index(drop=True),
            stats={"te_encoders": te_encoders},
        )
        # Share cat_encoders AND stats (stats carries mean/std + te_encoders so
        # val uses the same normalization AND fold-safe TE as train).
        val_ds = SpeedDataset(
            stops[test_mask].reset_index(drop=True),
            cat_encoders=train_ds.cat_encoders,
            stats=train_ds.get_stats(),
        )
        return train_ds, val_ds

    return SpeedDataset(stops), None


def load_temporal_occurrence_data(
    bins_path: Path,
    counts_path: Path,
    meta_path: Path,
    fold_idx: int = 0,
) -> Tuple[TemporalOccurrenceDataset, TemporalOccurrenceDataset]:
    """Load temporal occurrence data with walk-forward train/val split.

    Uses the same fold boundaries as the speed model to keep temporal
    alignment: train on years 0..N, validate on the next 6 months.
    """
    bins = pd.read_parquet(bins_path)
    for col in CELL_NUMERIC_FEATURES:
        if col not in bins.columns:
            bins[col] = 0.0

    # Walk-forward boundaries (matching config.py fold structure)
    fold_boundaries = [
        ("2018-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
        ("2018-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
        ("2018-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
        ("2018-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ]
    fold_idx = min(fold_idx, len(fold_boundaries) - 1)
    train_start, train_end, val_start, val_end = fold_boundaries[fold_idx]

    train_ds = TemporalOccurrenceDataset(
        counts_path, meta_path, bins,
        date_start=train_start, date_end=train_end,
    )
    val_ds = TemporalOccurrenceDataset(
        counts_path, meta_path, bins,
        date_start=val_start, date_end=val_end,
        stats=train_ds.get_stats(),
        nonzero_oversample=1.0,  # no oversampling for validation
    )
    return train_ds, val_ds
