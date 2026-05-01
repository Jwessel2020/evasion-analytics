#!/usr/bin/env python3
"""Stage 03 — Feature engineering across all 6 tiers.

Takes `data/stops_enriched.parquet` (Stage 02c output) and produces
`data/stops_features.parquet` with ~110 engineered features:

  - Tier 1: Time (cyclical, discrete, calendar, trend, shift)
  - Tier 2: Road geometry (from Stage 02a spatial join)
  - Tier 2b: POI distance (from Stage 02c distance join)
  - Tier 2c: Rolling / recency / trend (leakage-safe per-row windows)
  - Tier 3: Per-stop (vehicle, driver, detection, agency, violation)

Note: target encoding, local rolling statistics, and grid-based
aggregates are NOT computed here — they require per-fold fitting to
avoid leakage. Those go in Stage 05 training pipelines, which call
leakage-safe transformers during cross-validation.

Usage:
    cd ml/research
    python pipelines/03_feature_engineering.py
    python pipelines/03_feature_engineering.py --force

Output:
    data/stops_features.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# ---------- Tier 1: Time features ----------

# Full MD state holidays via python-holidays (12/year, 2018-2026).
import holidays as _holidays_pkg

_MD_HOLIDAYS = _holidays_pkg.US(years=range(2018, 2027), state="MD")

# Holiday categories — driven by enforcement character
# - dui_crackdown: big drinking holidays, DUI checkpoints surge
# - travel: interstate patrol on highways, speed traps
# - school_closed: different rush-hour pattern
# - regular: reduced general enforcement, federal holiday only
_HOLIDAY_CATEGORIES = {
    "New Year's Day": "dui_crackdown",
    "New Year's Day (observed)": "dui_crackdown",
    "Independence Day": "dui_crackdown",
    "Independence Day (observed)": "dui_crackdown",
    "Memorial Day": "travel",
    "Labor Day": "travel",
    "Thanksgiving Day": "travel",
    "Christmas Day": "travel",
    "Christmas Day (observed)": "travel",
    "Martin Luther King Jr. Day": "school_closed",
    "Inauguration Day; Martin Luther King Jr. Day": "school_closed",
    "Presidents' Day": "school_closed",
    "Columbus Day": "school_closed",
    "Veterans Day": "regular",
    "Veterans Day (observed)": "regular",
    "Juneteenth National Independence Day": "regular",
    "Juneteenth National Independence Day (observed)": "regular",
    "American Indian Heritage Day": "regular",
    "Inauguration Day": "regular",
}


def _build_holiday_date_sets():
    """Build fast-lookup sets for each holiday category + all holidays."""
    all_dates = set(_MD_HOLIDAYS.keys())
    dui_dates = {d for d, name in _MD_HOLIDAYS.items()
                 if _HOLIDAY_CATEGORIES.get(name) == "dui_crackdown"}
    travel_dates = {d for d, name in _MD_HOLIDAYS.items()
                    if _HOLIDAY_CATEGORIES.get(name) == "travel"}
    school_dates = {d for d, name in _MD_HOLIDAYS.items()
                    if _HOLIDAY_CATEGORIES.get(name) == "school_closed"}

    # Holiday weekends: Fri-Mon adjacent to a federal holiday
    weekend_dates = set()
    from datetime import timedelta
    for d in all_dates:
        dow = d.weekday()  # Mon=0..Sun=6
        # If holiday is Mon (0), the preceding Sat+Sun are part of weekend
        # If holiday is Fri (4), the following Sat+Sun
        # Always include the holiday itself
        weekend_dates.add(d)
        if dow == 0:  # Monday holiday
            weekend_dates.add(d - timedelta(days=1))  # Sun
            weekend_dates.add(d - timedelta(days=2))  # Sat
        elif dow == 4:  # Friday holiday
            weekend_dates.add(d + timedelta(days=1))  # Sat
            weekend_dates.add(d + timedelta(days=2))  # Sun
        elif dow == 3:  # Thursday holiday (e.g. Thanksgiving)
            weekend_dates.add(d + timedelta(days=1))  # Fri
            weekend_dates.add(d + timedelta(days=2))  # Sat
            weekend_dates.add(d + timedelta(days=3))  # Sun

    return all_dates, dui_dates, travel_dates, school_dates, weekend_dates


_ALL_H, _DUI_H, _TRAVEL_H, _SCHOOL_H, _WEEKEND_H = _build_holiday_date_sets()


def _days_to_nearest_holiday(d):
    """Min absolute days to nearest holiday (capped at 90)."""
    if not _ALL_H:
        return 90
    best = 90
    for h in _ALL_H:
        diff = abs((d - h).days)
        if diff < best:
            best = diff
            if best == 0:
                return 0
    return min(best, 90)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 1 — time-only features."""
    df = df.copy()

    # Parse timestamps
    df["stop_date"] = pd.to_datetime(df["stop_date"])
    # stop_time arrives as datetime.time or string — convert to hour decimal
    df["_stop_dt"] = pd.to_datetime(
        df["stop_date"].dt.strftime("%Y-%m-%d")
        + " "
        + df["stop_time"].astype(str)
    )

    # Raw integer bins (XGBoost can use either these or the cyclical ones)
    df["hour"] = df["_stop_dt"].dt.hour
    df["minute"] = df["_stop_dt"].dt.minute
    df["day_of_week"] = df["stop_date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["day_of_month"] = df["stop_date"].dt.day
    df["day_of_year"] = df["stop_date"].dt.dayofyear
    df["month"] = df["stop_date"].dt.month
    df["quarter"] = df["stop_date"].dt.quarter
    df["year"] = df["stop_date"].dt.year
    df["week_of_year"] = df["stop_date"].dt.isocalendar().week.astype("int")

    # Cyclical encodings — capture recurrence for tree models
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Contextual flags
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_rush_hour_morning"] = (
        (df["hour"].between(6, 9)) & (df["day_of_week"] < 5)
    ).astype("int8")
    df["is_rush_hour_evening"] = (
        (df["hour"].between(16, 19)) & (df["day_of_week"] < 5)
    ).astype("int8")
    df["is_night"] = (
        (df["hour"] >= 22) | (df["hour"] <= 5)
    ).astype("int8")
    df["is_late_night"] = (df["hour"] <= 4).astype("int8")
    df["is_school_zone_hour"] = (
        (
            df["hour"].between(6, 9) | df["hour"].between(14, 16)
        ) & (df["day_of_week"] < 5)
    ).astype("int8")
    df["is_bar_close_hour"] = (
        (df["hour"].between(1, 3)) & (df["day_of_week"].isin([4, 5]))
    ).astype("int8")

    # Calendar events — MD state holidays via python-holidays
    stop_date_d = df["stop_date"].dt.date
    df["is_us_federal_holiday"] = stop_date_d.isin(_ALL_H).astype("int8")
    df["is_dui_crackdown_holiday"] = stop_date_d.isin(_DUI_H).astype("int8")
    df["is_travel_holiday"] = stop_date_d.isin(_TRAVEL_H).astype("int8")
    df["is_school_closed_holiday"] = stop_date_d.isin(_SCHOOL_H).astype("int8")
    df["is_holiday_weekend"] = stop_date_d.isin(_WEEKEND_H).astype("int8")

    # Day-before/after holiday (often peak DUI + hungover driving)
    from datetime import timedelta
    _day_before = {d - timedelta(days=1) for d in _DUI_H}
    _day_after = {d + timedelta(days=1) for d in _DUI_H}
    df["is_day_before_dui_holiday"] = stop_date_d.isin(_day_before).astype("int8")
    df["is_day_after_dui_holiday"] = stop_date_d.isin(_day_after).astype("int8")

    # Days to nearest holiday (continuous, capped at 90)
    unique_dates = stop_date_d.drop_duplicates()
    date_to_days = {d: _days_to_nearest_holiday(d) for d in unique_dates}
    df["days_to_holiday"] = stop_date_d.map(date_to_days).astype("int16")

    df["days_until_weekend"] = (
        (4 - df["day_of_week"]).clip(lower=0)
    ).astype("int8")

    # Trend / regime features
    min_date = df["stop_date"].min()
    df["days_since_dataset_start"] = (
        (df["stop_date"] - min_date).dt.days.astype("int32")
    )
    df["is_covid_era"] = (
        (df["stop_date"] >= "2020-03-01") & (df["stop_date"] <= "2021-06-30")
    ).astype("int8")
    df["is_post_covid"] = (df["stop_date"] >= "2021-07-01").astype("int8")

    # Speculative: shift-change features assuming 8hr shifts starting 7/15/23
    df["hour_since_shift_start"] = (df["hour"] - 7) % 8
    # (not super useful alone but may catch fresh-shift patterns)

    # Drop internal column
    df = df.drop(columns=["_stop_dt"])

    return df


# ---------- Tier 2: Road geometry derived features ----------

def add_road_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 2 derived features. Direct road_* columns are already present
    from Stage 02a."""
    df = df.copy()

    # Speed egregiousness relative to the road's actual limit
    df["speed_over_ratio"] = (
        df["speed_over"] / df["road_maxspeed"]
    ).replace([np.inf, -np.inf], np.nan)

    df["is_excessive_speed"] = (df["speed_over"] > 20).astype("int8")
    df["is_extreme_speeding"] = (df["speed_over"] > 30).astype("int8")

    df["stop_on_curvy_road"] = (
        df["road_curvature"].fillna(0) > 200
    ).astype("int8")
    df["stop_on_mountain_pass"] = (
        df["road_mountain_pass"].fillna(False)
        | (df["road_max_grade"].fillna(0) > 8)
    ).astype("int8")

    # Normalized curvature per mile
    df["curvature_per_mi"] = (
        df["road_curvature"].fillna(0) / df["road_length_mi"].clip(lower=0.01)
    )

    return df


# ---------- Tier 2b: POI-derived features ----------
# Note: the distance features dist_*_m and proximity flags were already
# added at Stage 02c. This function only adds a few extra derived flags.

def add_poi_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra interaction flags built on top of the Stage 02c POI features."""
    df = df.copy()

    # School zone during school hours — highest-risk combo
    df["school_zone_during_hours"] = (
        df["is_in_school_zone"] & df["is_school_zone_hour"]
    ).astype("int8")

    # Late-night bar / mall proximity — DUI enforcement correlate
    df["urban_core_late_night"] = (
        df["is_in_urban_core"] & df["is_late_night"]
    ).astype("int8")

    # POI density proxy — inverse of average distance to top 3 POI categories
    # Small average = dense area, Large = rural
    df["avg_dist_top3_poi_m"] = df[
        ["dist_school_m", "dist_hospital_m", "dist_mall_m"]
    ].mean(axis=1)

    return df


# ---------- Tier 2c: Rolling / recency features ----------
#
# IMPORTANT: these are strictly time-causal. For a stop at time t, we only
# look at stops BEFORE t. The sort + expanding window in pandas handles this.
#
# These features are leakage-safe by construction, so they can live here
# in the static feature engineering stage. The rolling statistics DO mix
# train/val/test timestamps, but that's fine — at inference time, you only
# know history up to `now`, and the rolling features are computed the same way.
#
# The features we DON'T compute here (because they're leakage-prone):
#   - local_search_rate_{window} — needs per-fold target encoding
#   - agency_target_encoded — same
#
# Those go into the training pipelines as sklearn transformers.


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 2c rolling / recency features at the grid-cell level.

    Uses 3-decimal lat/lng rounding (~100m grid) as a fast proxy for H3.
    Time windows: 7 / 30 / 90 days. All windows are strictly causal
    (`closed='left'`) so the rolling stat never includes the current row
    or any future rows — leakage-safe.
    """
    df = df.copy()
    print("  Computing rolling features (can be slow on 300K+ rows) ...")

    # Grid id: simple rounding as a fast H3 proxy
    df["_grid_id"] = (
        df["latitude"].round(3).astype(str)
        + "_"
        + df["longitude"].round(3).astype(str)
    )

    # Sort by datetime ascending so expanding windows are causal
    df["_ts"] = pd.to_datetime(df["stop_date"])
    df = df.sort_values("_ts").reset_index(drop=True)

    # Dummy column of 1s — we'll sum it over time windows to get counts
    df["_one"] = 1

    # Compute rolling counts per grid cell using the time-indexed
    # groupby.rolling pattern. closed='left' → strict < t.
    for window_days in config.ROLLING_WINDOWS:
        col = f"stops_last_{window_days}d"
        print(f"    {col} ...")
        rolled = (
            df.set_index("_ts")
            .groupby("_grid_id")["_one"]
            .rolling(f"{window_days}D", closed="left")
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        # Re-align to the dataframe's current row order
        df[col] = rolled.to_numpy()
        df[col] = df[col].fillna(0).astype("int32")

    # Days since last stop in this cell (recency)
    # Within each grid, compute time delta from previous stop
    df["days_since_last_stop_here"] = (
        df.groupby("_grid_id")["_ts"]
        .diff()
        .dt.days
        .fillna(999)
        .astype("int32")
    )

    # Trending up: last 7d density > last 90d density? (guard against /0)
    df["stops_trending_up"] = (
        (df["stops_last_7d"] / 7) > (df["stops_last_90d"] / 90)
    ).astype("int8")

    df = df.drop(columns=["_grid_id", "_ts", "_one"])
    return df


# ---------- Tier 3: Per-stop derived features ----------

# ============================================================================
# Vehicle make normalization — the `vehicle_make` column has truncations
# (TOYT, HOND, CHEV, MERZ, VOLK, …) and variant spellings
# (MERCEDES / MERCEDES BENZ, VOLKSWAGON, INFINITY, LANDROVER). ~22% of all
# stops use a truncated or variant spelling. Without normalization the
# is_luxury flag misses ~37k Mercedes stops alone (3% of the dataset),
# and county-baseline vehicle-mix stats get fragmented. Audit 2026-04-24.
# ============================================================================

VEHICLE_MAKE_ALIASES = {
    # Mercedes family (~37k stops mistagged pre-fix)
    "MERZ": "MERCEDES-BENZ",
    "MERC": "MERCEDES-BENZ",
    "MERCEDES": "MERCEDES-BENZ",
    "MERCEDEZ": "MERCEDES-BENZ",
    "MERCEDES BENZ": "MERCEDES-BENZ",
    # High-volume truncations (common brands)
    "TOYT": "TOYOTA", "TOYO": "TOYOTA",
    "HOND": "HONDA",
    "CHEV": "CHEVROLET", "CHEVY": "CHEVROLET",
    "NISS": "NISSAN",
    "HYUN": "HYUNDAI",
    "ACUR": "ACURA",
    "DODG": "DODGE",
    "SUBA": "SUBARU",
    "CHRY": "CHRYSLER",
    "MITS": "MITSUBISHI",
    "MAZD": "MAZDA",
    "CADI": "CADILLAC",
    "INFI": "INFINITI", "INFINITY": "INFINITI",
    "BUIC": "BUICK",
    "LEXS": "LEXUS", "LEXU": "LEXUS",
    "VOLV": "VOLVO",
    "PONT": "PONTIAC",
    "LINC": "LINCOLN",
    # Volkswagen variants (~24k across forms)
    "VW": "VOLKSWAGEN",
    "VOLK": "VOLKSWAGEN",
    "VOLKS": "VOLKSWAGEN",
    "VOLKSWAGON": "VOLKSWAGEN",  # common misspelling
    # Land Rover variants
    "LANDROVER": "LAND ROVER",
    "LNDR": "LAND ROVER",
    "RANGE ROVER": "LAND ROVER",
    # Supercar variants
    "ROLLS ROYCE": "ROLLS-ROYCE",
    "ROLLSROYCE": "ROLLS-ROYCE",
    "ASTON": "ASTON MARTIN",
    "ASTONMARTIN": "ASTON MARTIN",
    "ALFA": "ALFA ROMEO",
    "ALFAROMEO": "ALFA ROMEO",
    # Motorcycle variants (mostly for proper counting)
    "HARL": "HARLEY-DAVIDSON",
    "HARLEY": "HARLEY-DAVIDSON",
    "HARLEY DAVIDSON": "HARLEY-DAVIDSON",
    # Commercial truck
    "FRGT": "FREIGHTLINER", "FREIGHT": "FREIGHTLINER",
}


# Expanded luxury — original 14 + supercars + modern EV luxury + heritage.
# Matched against the NORMALIZED make (after VEHICLE_MAKE_ALIASES), exact.
LUXURY_MAKES = {
    # Traditional luxury
    "MERCEDES-BENZ", "BMW", "AUDI", "LEXUS", "PORSCHE",
    "CADILLAC", "INFINITI", "ACURA", "JAGUAR", "LAND ROVER",
    "LINCOLN", "MASERATI", "BENTLEY", "TESLA",
    # Supercars — an enthusiast app CANNOT miss these
    "FERRARI", "LAMBORGHINI", "MCLAREN", "ROLLS-ROYCE",
    "ASTON MARTIN", "LOTUS", "BUGATTI", "PAGANI", "KOENIGSEGG",
    # Modern luxury / EV
    "GENESIS", "POLESTAR", "LUCID", "RIVIAN",
    # Heritage
    "ALFA ROMEO",
}


# Makes where EVERY model is performance — no model-string lookup needed.
# Useful for supercars that may appear with obscure model names.
PERFORMANCE_MAKES_OUTRIGHT = {
    "FERRARI", "LAMBORGHINI", "MCLAREN", "BUGATTI",
    "PAGANI", "KOENIGSEGG", "LOTUS",
}


# Motorcycle-only brands. Mixed brands (HONDA, YAMAHA, KAWASAKI, BMW, SUZUKI)
# require vehicle_type to confirm — not listed here.
MOTORCYCLE_MAKES = {
    "HARLEY-DAVIDSON", "DUCATI", "TRIUMPH", "KTM", "INDIAN",
    "MV AGUSTA", "MOTO GUZZI", "APRILIA", "HUSQVARNA",
}


# Expanded performance model tokens. Matched via substring on vehicle_model.
# Critical: use SPECIFIC tokens (M3, RS3, GT3) — bare "M" / "RS" match too
# broadly and produce false positives on common models.
PERFORMANCE_MODELS = {
    # Classic American muscle + JDM staples
    "CORVETTE", "MUSTANG", "CAMARO", "CHALLENGER", "CHARGER",
    "WRX", "STI", "TYPE R", "FOCUS RS", "SUPRA", "GT-R",
    "911",
    # BMW M (specific, not bare "M")
    "M2", "M3", "M4", "M5", "M6", "M8",
    "M235", "M240", "M340", "M440", "M550", "M760",
    # Audi RS/S (specific, not bare "RS")
    "RS3", "RS4", "RS5", "RS6", "RS7", "R8", "TT RS",
    "S3 ", "S4 ", "S5 ", "S6 ", "S7 ", "S8 ",  # trailing space avoids CAMRY S etc.
    # Mercedes AMG sub-models (bare AMG kept as a catch-all)
    "AMG",
    "C63", "E63", "G63", "S63", "GT63", "GT 63",
    "A45", "CLA45", "GLA45", "GLC63", "C43", "E53",
    # Porsche performance
    "GT2", "GT3", "GT4", "TURBO S", "CAYMAN", "BOXSTER",
    "718", "TAYCAN",
    # Dodge / SRT
    "HELLCAT", "DEMON", "SRT", "REDEYE", "SCAT PACK",
    "TRACKHAWK", "TRX", "VIPER",
    # Ford performance
    "RAPTOR", "SHELBY", "GT350", "GT500", "BOSS 302", "BOSS 429",
    # Tesla performance
    "PLAID",
    # Ferrari
    "488", "458", "F8", "SF90", "ROMA", "PORTOFINO",
    "296", "812", "TESTAROSSA", "LAFERRARI",
    # Lamborghini
    "HURACAN", "AVENTADOR", "URUS", "REVUELTO", "GALLARDO",
    "MURCIELAGO", "COUNTACH",
    # McLaren
    "570S", "720S", "765LT", "SENNA", "ARTURA",
    # Nissan sport
    "370Z", "400Z", "GTR", "R32", "R33", "R34", "R35",
    # Hyundai N (enthusiast sub-brand)
    "VELOSTER N", "ELANTRA N", "KONA N",
    # Genesis
    "G70", "G80", "G90",
    # Acura performance
    "NSX", "TYPE S",
    # Lexus F
    "LFA", "RC F", "IS F", "GS F",
    # Rolls / Bentley sport
    "CONTINENTAL GT", "FLYING SPUR",
    # Aston Martin
    "VANTAGE", "DB11", "DB12", "DBS", "DB9", "VANQUISH",
    # Alfa Romeo performance
    "GIULIA QV", "STELVIO QV",
}


COMMERCIAL_TYPES = {
    "TRUCK", "TRACTOR", "BUS", "TAXI", "TRAILER", "COMMERCIAL",
}

NEIGHBOR_STATES = {"VA", "DC", "WV", "PA", "DE"}


# ============================================================================
# Arrest-type indicators — the `arrest_type` column encodes enforcement
# method (NOT whether an arrest occurred — see note at bottom of this file).
# Codes have the form "A - Marked Patrol", "Q - Marked Laser", etc.
# Two orthogonal axes matter for both speed and trap prediction:
#   (1) stationary vs mobile:   laser/radar guns vs moving patrol cars
#   (2) marked vs unmarked:     visible cruiser vs stealth vehicle
# ============================================================================

# Letter-code → axis classification
_STATIONARY_CODES = frozenset({"E", "F", "G", "H", "Q", "R", "S"})
# A - Marked Patrol, B - Unmarked Patrol, C/D - VASCAR, I/J - Moving Radar (Moving)
_MOBILE_CODES = frozenset({"A", "B", "C", "D", "I", "J"})
_MARKED_CODES = frozenset({"A", "C", "E", "G", "I", "L", "M", "P", "Q"})
_UNMARKED_CODES = frozenset({"B", "D", "F", "H", "J", "N", "R"})
_AUTOMATED_CODES = frozenset({"S"})  # LPR (license plate recognition)


def add_per_stop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 3 — vehicle, driver, detection, agency, violation features."""
    df = df.copy()

    # ---- Vehicle ----
    df["vehicle_age"] = (df["year"] - df["vehicle_year"]).clip(
        lower=0, upper=60
    )
    # Median imputation for missing vehicle_year
    med_age = df["vehicle_age"].median()
    df["vehicle_age"] = df["vehicle_age"].fillna(med_age)

    def _contains_any(series: pd.Series, words: set) -> pd.Series:
        s = series.fillna("").str.upper()
        return pd.Series(
            [any(w in x for w in words) for x in s],
            index=s.index,
        ).astype("int8")

    df["is_truck"] = _contains_any(df["vehicle_type"], {"TRUCK"})
    df["is_motorcycle"] = _contains_any(df["vehicle_type"], {"MOTORCYCLE", "CYCLE"})
    df["is_suv"] = _contains_any(df["vehicle_type"], {"SUV", "SPORT UTILITY"})
    df["is_sedan"] = _contains_any(df["vehicle_type"], {"SEDAN", "PASSENGER"})
    df["is_van"] = _contains_any(df["vehicle_type"], {"VAN"})
    df["is_commercial"] = _contains_any(df["vehicle_type"], COMMERCIAL_TYPES)

    # ---- Vehicle make normalization ----
    # Collapse truncations (TOYT, MERZ, …) and variant spellings to canonical
    # names BEFORE any luxury/performance tagging. Stored as new column so
    # the raw `vehicle_make` remains unchanged for auditability.
    _mk = df["vehicle_make"].fillna("").str.upper().str.strip()
    df["vehicle_make_norm"] = _mk.map(VEHICLE_MAKE_ALIASES).fillna(_mk)

    df["is_luxury"] = df["vehicle_make_norm"].isin(LUXURY_MAKES).astype("int8")

    # is_performance = (entire-brand performance) OR (model matches a known
    # performance token). Supercars are flagged purely by make; everything
    # else goes through the expanded PERFORMANCE_MODELS token list.
    is_perf_make = df["vehicle_make_norm"].isin(PERFORMANCE_MAKES_OUTRIGHT)
    is_perf_model = _contains_any(df["vehicle_model"], PERFORMANCE_MODELS)
    df["is_performance"] = (is_perf_make | is_perf_model.astype(bool)).astype("int8")

    # Motorcycle by make (orthogonal signal from vehicle_type-based
    # is_motorcycle above — catches Harley/Ducati/etc. without needing
    # vehicle_type to be set).
    df["is_motorcycle_brand"] = df["vehicle_make_norm"].isin(
        MOTORCYCLE_MAKES
    ).astype("int8")

    df["vehicle_year_decade"] = (
        (df["vehicle_year"] // 10 * 10).fillna(0).astype("int32")
    )

    # ---- Driver ----
    df["is_out_of_state"] = (
        (df["driver_state"].fillna("MD") != "MD")
    ).astype("int8")
    df["is_neighbor_state"] = df["driver_state"].isin(
        NEIGHBOR_STATES
    ).astype("int8")
    df["is_far_state"] = (
        df["is_out_of_state"] & ~df["is_neighbor_state"]
    ).astype("int8")

    # ---- Detection method ----
    # One-hot the radar/laser/vascar/patrol encoding
    for method in ["radar", "laser", "vascar", "patrol"]:
        df[f"detection_method_{method}"] = (
            df["detection_method"].fillna("").str.lower() == method
        ).astype("int8")
    df["detection_method_unknown"] = (
        df["detection_method"].isna()
        | (df["detection_method"].fillna("").str.lower() == "unknown")
    ).astype("int8")

    # ---- Arrest-type indicators (marked/unmarked × stationary/mobile) ----
    # `arrest_type` values look like "A - Marked Patrol", "Q - Marked Laser".
    # The leading single letter is the enforcement method code. Marked/
    # unmarked is ORTHOGONAL to stationary/mobile — both axes predict
    # separately for enthusiasts (unmarked stealth enforcement matters for
    # both mobile patrol AND stationary radar).
    _at_code = df["arrest_type"].fillna("").str.extract(
        r"^([A-Z])\s*-", expand=False
    ).fillna("").str.upper()
    df["is_stationary_trap"] = _at_code.isin(_STATIONARY_CODES).astype("int8")
    df["is_mobile_enforcement"] = _at_code.isin(_MOBILE_CODES).astype("int8")
    df["is_marked_unit"] = _at_code.isin(_MARKED_CODES).astype("int8")
    df["is_unmarked_unit"] = _at_code.isin(_UNMARKED_CODES).astype("int8")
    df["is_automated_detection"] = _at_code.isin(_AUTOMATED_CODES).astype("int8")
    # v3.3.0: preserve the raw letter as a categorical feature so the FT-T
    # can learn embeddings that distinguish A vs M vs L (all currently
    # is_marked_unit=1 but distinct in the data — Marked Patrol vs
    # Off-Duty vs Motorcycle). 15 letters observed in current data;
    # cardinality cap = 16 in deep/data.py STOP_CAT_CARDINALITIES.
    df["arrest_type_letter"] = _at_code.where(_at_code != "", "UNK")

    # v3.3.0: disposition target — 1 if Citation else 0. Used by the new
    # disposition head in deep/model.py. Note: violation_type column is
    # SIMULTANEOUSLY removed from STOP_CAT_FEATURES (deep/data.py) to
    # prevent target leakage. ~66% positive rate in current data.
    df["is_citation"] = (df["violation_type"] == "Citation").astype("int8")

    # ---- Agency ----
    # Coarse agency type classification (state / county / municipal / sheriff / tap)
    def _classify_agency(a: str | None) -> str:
        if not isinstance(a, str):
            return "unknown"
        a = a.upper()
        if "STATE" in a or "TROOPER" in a:
            return "state_police"
        if "SHERIFF" in a:
            return "sheriff"
        if "MTAP" in a or "TRANSPORTATION AUTHORITY" in a:
            return "mtap"
        if "COUNTY" in a:
            return "county"
        if "CITY" in a or "POLICE" in a:
            return "municipal"
        return "other"

    df["agency_type"] = df["agency"].apply(_classify_agency)
    df["agency_is_traffic_focused"] = df["agency_type"].isin(
        {"state_police", "mtap"}
    ).astype("int8")

    # ---- Violation ----
    # Top 20 violation types, everything else → "other"
    top_vt = df["violation_type"].value_counts().nlargest(20).index
    df["violation_type_top"] = df["violation_type"].where(
        df["violation_type"].isin(top_vt), other="other"
    )

    return df


# ---------- Orchestrator ----------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all tiers in order. Prints timing per tier."""
    print("\n[Tier 1] Time features ...")
    t0 = time.time()
    df = add_time_features(df)
    print(f"  Done in {time.time()-t0:.1f}s — {len(df.columns)} columns total")

    print("\n[Tier 2] Road-derived features ...")
    t0 = time.time()
    df = add_road_derived_features(df)
    print(f"  Done in {time.time()-t0:.1f}s — {len(df.columns)} columns total")

    print("\n[Tier 2b] POI-derived features ...")
    t0 = time.time()
    df = add_poi_derived_features(df)
    print(f"  Done in {time.time()-t0:.1f}s — {len(df.columns)} columns total")

    print("\n[Tier 2c] Rolling / recency features ...")
    t0 = time.time()
    df = add_rolling_features(df)
    print(f"  Done in {time.time()-t0:.1f}s — {len(df.columns)} columns total")

    print("\n[Tier 3] Per-stop features ...")
    t0 = time.time()
    df = add_per_stop_features(df)
    print(f"  Done in {time.time()-t0:.1f}s — {len(df.columns)} columns total")

    # NOTE: the upstream `arrest_type` column in the MCP dataset turned
    # out to be the detection/enforcement method (values like "A - Marked
    # Patrol", "Q - Marked Laser"), not whether an arrest occurred. There
    # is no real arrest-outcome signal in the data — `arrest_made` was
    # removed from OUTCOME_TARGETS in config.py after Stage 05 exposed
    # this as 100% positive. See config.OUTCOME_TARGETS comment.

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output parquet",
    )
    args = parser.parse_args()

    in_path = config.PATH_STOPS_ENRICHED
    out_path = config.PATH_STOPS_FEATURES

    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run Stage 02c first.")
        return 2

    if out_path.exists() and not args.force:
        print(f"ERROR: {out_path} already exists. Use --force to overwrite.")
        return 2

    print("=" * 60)
    print("Stage 03 — Feature engineering (6 tiers, ~110 features)")
    print("=" * 60)

    print(f"  Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    df_feat = build_features(df)

    print(f"\n  Writing parquet → {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_path, compression="zstd", index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {size_mb:.1f} MB")
    print(f"  Final shape: {len(df_feat):,} × {len(df_feat.columns)}")
    print("\nStage 03 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
