"""
preprocess.py
Full preprocessing pipeline for Norwegian electricity price forecasting.

Steps:
  1.  Filter to hourly rows only
  2.  Drop useless columns (solar, offshore wind, date)
  3.  Forward-fill fuels and USD_EUR (weekend/holiday gaps)
  4.  Convert USD fuel prices to EUR
  5.  Impute wind and load nulls
  6.  Drop rows where Price is null (target — cannot impute)
  7.  Convert UTC timestamps to CET
  8.  Create time features from CET (plain integers)
  9.  Create lag features (all causal, all EUR)
  10. Drop raw columns replaced by lags
  11. Drop warmup rows (first 168 rows per zone)
  12. Add split column (train / val / test)
  13. Assert zero nulls in final dataset
  14. Save to outputs/processed_dataset.csv

Run: python preprocess.py
"""

import pandas as pd
import numpy as np
from config import (
    RAW_CSV, OUTPUT_DIR, PROCESSED_CSV,
    ZONES, TARGET, TIMEZONE,
    COLS_DROP, COLS_USD_FUEL, COL_EUR_FUEL, COL_FX,
    COLS_FFILL_FUEL, COLS_LOAD, COLS_WIND, WIND_RENAME,
    ZONES_NO_WIND, WARMUP_ROWS, FFILL_LIMIT_WIND, FFILL_LIMIT_LOAD,
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Load raw CSV and filter to hourly rows ────────────────────────────
def load_and_filter_hourly(path) -> pd.DataFrame:
    """
    Load the raw CSV and keep only on-the-hour rows.
    The dataset switches to 15-minute resolution from March 2025 onward
    due to an ENTSO-E format change. Those sub-hourly rows carry no
    day-ahead price and corrupt lag features, so we drop them.
    """
    print("Step 1: Loading data and filtering to hourly rows...")
    df = pd.read_csv(path, parse_dates=["time_utc"])
    before = len(df)
    df = df[df["time_utc"].dt.minute == 0].copy()
    after = len(df)
    print(f"  Rows before: {before:,}  after hourly filter: {after:,}  "
          f"(dropped {before - after:,} sub-hourly rows)")
    return df


# ── Step 2: Drop useless columns ─────────────────────────────────────────────
def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that carry no signal for this forecasting task:
    - Solar: Norway has negligible solar capacity (96-99% null)
    - Offshore wind: nearly entirely absent (99%+ null)
    - date: redundant once time_utc is available
    """
    print("Step 2: Dropping useless columns...")
    df = df.drop(columns=COLS_DROP, errors="ignore")
    print(f"  Remaining columns: {list(df.columns)}")
    return df


# ── Step 3: Forward-fill fuels and USD_EUR per zone ──────────────────────────
def ffill_fuels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuel and FX markets are closed on weekends and public holidays.
    This produces structured gaps (~28-30% of rows, always Saturday/Sunday).
    Forward-fill carries the last known price forward — exactly what a real
    forecaster would use on a weekend.

    Must be done BEFORE EUR conversion because:
    gas_eur = gas_usd × usd_eur
    If both are NaN on weekends, multiplication gives NaN even after filling.
    """
    print("Step 3: Forward-filling fuel and FX columns per zone...")
    df = df.sort_values(["zone", "time_utc"]).reset_index(drop=True)
    for col in COLS_FFILL_FUEL:
        if col in df.columns:
            df[col] = df.groupby("zone")[col].ffill()
    nulls = df[COLS_FFILL_FUEL].isnull().sum().sum()
    print(f"  Nulls remaining in fuel/FX columns after ffill: {nulls}")
    return df


# ── Step 4: Convert USD fuel prices to EUR ───────────────────────────────────
def convert_to_eur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gas, coal and oil are priced in USD. The model predicts EUR/MWh prices.
    Converting to EUR makes all features and the target the same currency,
    removing the need for the model to implicitly learn the FX relationship.

    EUA (carbon) is already in EUR — just rename it.
    USD_EUR is dropped after conversion — its information is now embedded
    in the converted fuel prices.
    """
    print("Step 4: Converting USD fuel prices to EUR...")
    df = df.copy()

    # gas_fM_01 → gas_eur, coal_fM_01 → coal_eur, oil_fM_01 → oil_eur
    fuel_name_map = {
        "gas_fM_01":  "gas_eur",
        "coal_fM_01": "coal_eur",
        "oil_fM_01":  "oil_eur",
    }
    for usd_col, eur_col in fuel_name_map.items():
        df[eur_col] = df[usd_col] * df[COL_FX]
        print(f"  {usd_col} (USD) × {COL_FX} → {eur_col} (EUR)")

    # EUA is already EUR — just rename
    df["eua_eur"] = df[COL_EUR_FUEL]

    # Drop original USD columns and FX rate
    drop_cols = COLS_USD_FUEL + [COL_EUR_FUEL, COL_FX]
    df = df.drop(columns=drop_cols)
    print(f"  Dropped original: {drop_cols}")
    return df


# ── Step 5: Impute wind and load nulls ────────────────────────────────────────
def impute_wind_and_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load columns have <0.1% nulls — isolated reporting gaps.
    Wind onshore has 2-20% nulls for NO1-NO4, gaps up to 48h.
    NO5 has 100% null wind — genuinely no wind generation reported.

    Strategy:
    - Load:      forward-fill within zone, limit=3 hours
    - Wind NO1-NO4: forward-fill within zone, limit=48 hours
    - Wind NO5:  fill with 0 (no wind capacity exists)
    """
    print("Step 5: Imputing wind and load nulls...")

    # Rename wind columns to shorter names first
    df = df.rename(columns=WIND_RENAME)
    wind_actual   = "WindOnshore_Actual"
    wind_dayahead = "WindOnshore_DayAhead"

    for zone in ZONES:
        mask = df["zone"] == zone

        # Load: forward-fill limit=3
        for col in COLS_LOAD:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col].ffill(limit=FFILL_LIMIT_LOAD)

        # Wind
        for col in [wind_actual, wind_dayahead]:
            if col in df.columns:
                if zone in ZONES_NO_WIND:
                    df.loc[mask, col] = df.loc[mask, col].fillna(0)
                else:
                    df.loc[mask, col] = df.loc[mask, col].ffill(limit=FFILL_LIMIT_WIND)

    for col in ["Load_Actual", "Load_DayAhead", wind_actual, wind_dayahead]:
        if col in df.columns:
            remaining = df[col].isnull().sum()
            print(f"  {col:35s} nulls remaining: {remaining}")
    return df


# ── Step 6: Drop Price null rows ──────────────────────────────────────────────
def drop_price_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price is the target variable. Imputing it would mean training the model
    on fabricated labels. Drop all rows where Price is null.
    """
    print("Step 6: Dropping rows where Price is null...")
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    after = len(df)
    print(f"  Dropped {before - after:,} rows ({(before-after)/before*100:.1f}%)")
    return df


# ── Step 7: Convert UTC to CET ────────────────────────────────────────────────
def convert_to_cet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nord Pool operates in CET (UTC+1 winter) / CEST (UTC+2 summer).
    Time features extracted from UTC show a shifting peak hour across DST.
    Extracting from CET gives a consistent 08:00-09:00 morning peak all year.
    Both timestamps are kept in the CSV for reference.
    """
    print("Step 7: Converting UTC to CET...")
    df["time_cet"] = (
        df["time_utc"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(TIMEZONE)
    )
    print(f"  Winter example: UTC {df['time_utc'].iloc[0]} "
          f"→ CET {df['time_cet'].iloc[0]}")
    return df


# ── Step 8: Create time features ─────────────────────────────────────────────
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plain integer time features extracted from CET timestamps.
    No sin/cos encoding — plain integers are human readable.
    Tree models handle integer time features natively.
    LSTM learns temporal patterns from the sequence directly.
    """
    print("Step 8: Creating time features from CET...")
    df["hour"]        = df["time_cet"].dt.hour
    df["day_of_week"] = df["time_cet"].dt.day_of_week   # 0=Monday, 6=Sunday
    df["month"]       = df["time_cet"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_winter"]   = df["month"].isin([12, 1, 2]).astype(int)
    print("  Created: hour, day_of_week, month, is_weekend, is_winter")
    return df


# ── Step 9: Create lag features ───────────────────────────────────────────────
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All lag features are computed per zone independently so that zone
    boundaries do not bleed into each other's lag windows.

    Causality rules:
    - price_lag24 / price_lag168: safe — auction results from yesterday/last week
    - load_actual_lag1: real-time load published with 1h delay (ENTSO-E rule)
    - wind_actual_lag1: real-time wind published with 1h delay
    - fuel_lag1d: daily settlement price from the previous day (published ~18:00)
    """
    print("Step 9: Creating lag features per zone...")
    df = df.sort_values(["zone", "time_utc"]).reset_index(drop=True)

    lag_definitions = {
        "price_lag24":      (TARGET,               24),
        "price_lag168":     (TARGET,              168),
        "load_actual_lag1": ("Load_Actual",          1),
        "wind_actual_lag1": ("WindOnshore_Actual",   1),
        "gas_lag1d":        ("gas_eur",             24),
        "coal_lag1d":       ("coal_eur",            24),
        "oil_lag1d":        ("oil_eur",             24),
        "eua_lag1d":        ("eua_eur",             24),
    }

    for new_col, (source_col, shift) in lag_definitions.items():
        df[new_col] = df.groupby("zone")[source_col].shift(shift)
        print(f"  {new_col:20s} ← {source_col} shifted {shift:3d} rows")

    return df


# ── Step 10: Drop raw columns replaced by lags ───────────────────────────────
def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the unlagged versions of columns now represented by lagged features.
    Keeping them would risk the model accidentally using unlagged (future) values.
    Load_DayAhead and WindOnshore_DayAhead are kept — they are legitimately
    available at forecast time (published d-1 at 08:00 before the auction).
    """
    print("Step 10: Dropping raw columns replaced by lags...")
    cols_to_drop = [
        "Load_Actual",
        "WindOnshore_Actual",
        "gas_eur",
        "coal_eur",
        "oil_eur",
        "eua_eur",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"  Dropped: {cols_to_drop}")
    print(f"  Kept:    Load_DayAhead, WindOnshore_DayAhead (available d-1 at 08:00)")
    return df


# ── Step 11: Drop warmup rows ─────────────────────────────────────────────────
def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    After shifting price_lag168 by 168 rows, the first 168 rows per zone
    are NaN because there is no week-prior data at the start of the dataset.

    168 rows = 7 days. After dropping:
    - price_lag24  is guaranteed valid (needs only 24 rows of history)
    - price_lag168 is guaranteed valid (needs 168 rows of history)

    Loss: 7 days × 5 zones = 840 rows out of ~260,000 (less than 0.3%)

    Note: pandas 3.x groupby drops the grouping column from the result.
    Using a manual loop to avoid this.
    """
    print(f"Step 11: Dropping first {WARMUP_ROWS} warmup rows per zone...")
    before = len(df)
    parts = []
    for zone in ZONES:
        part = df[df["zone"] == zone].iloc[WARMUP_ROWS:]
        parts.append(part)
    df = pd.concat(parts).sort_values(["zone", "time_utc"]).reset_index(drop=True)
    after = len(df)
    print(f"  Dropped {before - after:,} warmup rows "
          f"({WARMUP_ROWS} per zone × {len(ZONES)} zones)")
    return df


# ── Step 12: Add split column ─────────────────────────────────────────────────
def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row with its split using the year of time_utc.
    Using end-of-day timestamps ensures the full last day of each
    period is correctly captured.

    Train      = 2020–2023
    Validation = 2024
    Test       = 2025
    """
    print("Step 12: Adding split column...")
    conditions = [
        df["time_utc"] <= TRAIN_END,
        (df["time_utc"] >= VAL_START) & (df["time_utc"] <= VAL_END),
        df["time_utc"] >= TEST_START,
    ]
    df["split"] = np.select(conditions, ["train", "val", "test"], default="unknown")
    counts = df["split"].value_counts().to_dict()
    for split in ["train", "val", "test", "unknown"]:
        print(f"  {split:8s}: {counts.get(split, 0):,} rows")
    if counts.get("unknown", 0) > 0:
        raise ValueError("Unknown split rows found — check TRAIN_END/VAL_START boundaries.")
    return df


# ── Step 13: Final null check ─────────────────────────────────────────────────
def assert_no_nulls(df: pd.DataFrame):
    """
    Hard check — if any nulls remain raise an error immediately.
    The final processed CSV must be completely clean.
    """
    print("Step 13: Asserting zero nulls...")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        raise ValueError(
            f"Nulls found in processed dataset:\n{null_counts.to_string()}\n"
            "Fix the preprocessing pipeline before proceeding."
        )
    print("  Zero nulls confirmed.")


# ── Step 14: Reorder columns and save ────────────────────────────────────────
def reorder_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arrange columns in a clean logical order.
    Target (Price) is always the last column.
    """
    print("Step 14: Reordering columns and saving...")

    final_cols = [
        "time_utc", "time_cet", "zone", "split",
        "hour", "day_of_week", "month", "is_weekend", "is_winter",
        "price_lag24", "price_lag168",
        "Load_DayAhead", "load_actual_lag1",
        "WindOnshore_DayAhead", "wind_actual_lag1",
        "gas_lag1d", "coal_lag1d", "oil_lag1d", "eua_lag1d",
        TARGET,
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    df.to_csv(PROCESSED_CSV, index=False)
    print(f"  Saved → {PROCESSED_CSV}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_and_filter_hourly(RAW_CSV)
    df = drop_useless_columns(df)
    df = ffill_fuels(df)
    df = convert_to_eur(df)
    df = impute_wind_and_load(df)
    df = drop_price_nulls(df)
    df = convert_to_cet(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = drop_raw_columns(df)
    df = drop_warmup_rows(df)
    df = add_split_column(df)
    assert_no_nulls(df)
    df = reorder_and_save(df)

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total rows   : {len(df):,}")
    print(f"  Total columns: {df.shape[1]}")
    print(f"  Zones        : {df['zone'].unique().tolist()}")
    print()
    print("  Split breakdown:")
    for split, count in df.groupby("split")["split"].count().items():
        pct = count / len(df) * 100
        print(f"    {split:8s}: {count:,} rows ({pct:.1f}%)")
    print()
    print("  Date range:")
    print(f"    From : {df['time_utc'].min()}")
    print(f"    To   : {df['time_utc'].max()}")
    print()
    print("  Final columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    return df


if __name__ == "__main__":
    run()