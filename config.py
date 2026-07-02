"""
config.py
Single source of truth for all constants used across the pipeline.
Edit here — changes propagate to preprocess.py, data_loader.py,
benchmarks.py, model.py, lstm_model.py and evaluate.py.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent
RAW_CSV        = ROOT / "NO1_NO5_2020_2025_raw_data.csv"
OUTPUT_DIR     = ROOT / "outputs"
PROCESSED_CSV  = OUTPUT_DIR / "processed_dataset.csv"
MODEL_DIR      = OUTPUT_DIR / "models"
PLOT_DIR       = OUTPUT_DIR / "plots"

# ── Zones ──────────────────────────────────────────────────────────────────────
ZONES         = ["NO1", "NO2", "NO3", "NO4", "NO5"]
ZONES_NO_WIND = ["NO5"]   # NO5 reports no onshore wind generation at all

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET = "Price"

# ── Raw column groups (from NO1_NO5_2020_2025_raw_data.csv, 17 cols) ───────────
# Raw columns: time_utc, Price, Load_Actual, Load_DayAhead,
#   Generation_Solar_Actual, Generation_Solar_DayAhead,
#   Generation_WindOffshore_DayAhead, Generation_WindOnshore_Actual,
#   Generation_WindOnshore_DayAhead, zone, Generation_WindOffshore_Actual,
#   date, EUA_fM_01, USD_EUR, coal_fM_01, gas_fM_01, oil_fM_01

# Step 2 — dropped: solar (96–100% null), offshore wind (negligible), date (redundant)
COLS_DROP = [
    "Generation_Solar_Actual",
    "Generation_Solar_DayAhead",
    "Generation_WindOffshore_DayAhead",
    "Generation_WindOffshore_Actual",
    "date",
]

# Step 3 — forward-filled (weekend/holiday market closures)
COLS_FFILL_FUEL = ["EUA_fM_01", "USD_EUR", "coal_fM_01", "gas_fM_01", "oil_fM_01"]

# Step 4 — currency conversion
COLS_USD_FUEL = ["gas_fM_01", "coal_fM_01", "oil_fM_01"]   # priced in USD
COL_EUR_FUEL  = "EUA_fM_01"                                 # already EUR
COL_FX        = "USD_EUR"

# Step 5 — load / wind imputation
COLS_LOAD = ["Load_Actual", "Load_DayAhead"]
COLS_WIND = ["Generation_WindOnshore_Actual", "Generation_WindOnshore_DayAhead"]
WIND_RENAME = {
    "Generation_WindOnshore_Actual":   "WindOnshore_Actual",
    "Generation_WindOnshore_DayAhead": "WindOnshore_DayAhead",
}

FFILL_LIMIT_WIND = 48   # max hours to forward-fill wind gaps (NO1–NO4)
FFILL_LIMIT_LOAD = 3    # max hours to forward-fill load gaps

# Step 11 — warmup rows dropped per zone (price_lag168 needs 168h of history)
WARMUP_ROWS = 168

# ── Model feature set (processed_dataset.csv, 20 cols total) ───────────────────
# Excludes time_utc, time_cet, zone, split, Price (target) → 15 features
FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_winter",
    "price_lag24",
    "price_lag168",
    "Load_DayAhead",
    "load_actual_lag1",
    "WindOnshore_DayAhead",
    "wind_actual_lag1",
    "gas_lag1d",
    "coal_lag1d",
    "oil_lag1d",
    "eua_lag1d",
]

# ── Timezone ───────────────────────────────────────────────────────────────────
TIMEZONE = "Europe/Oslo"   # CET (UTC+1) / CEST (UTC+2)

# ── Main train/val/test split (single split, used by data_loader.get_splits) ──
TRAIN_END  = "2023-12-31 23:59:59"
VAL_START  = "2024-01-01 00:00:00"
VAL_END    = "2024-12-31 23:59:59"
TEST_START = "2025-01-01 00:00:00"
TEST_END   = "2025-12-31 23:59:59"

# ── Rolling window splits (used by evaluate.py) ─────────────────────────────────
# FIX: previously val_year == test_year for every window, which meant LSTM
# early-stopping and MLP GridSearchCV were tuned directly on the test set
# (data leakage). Corrected to a strict walk-forward scheme: for every window,
# train < val < test, all three are distinct, non-overlapping calendar years,
# and expansion happens one year at a time.
#
#   Window 1: train=2020,      val=2021, test=2022
#   Window 2: train=2020–2021, val=2022, test=2023
#   Window 3: train=2020–2022, val=2023, test=2024
#   Window 4: train=2020–2023, val=2024, test=2025   ← matches the main split above
ROLLING_WINDOWS = [
    {
        "name":       "Window 1",
        "train_end":  "2020-12-31 23:59:59",
        "val_start":  "2021-01-01 00:00:00",
        "val_end":    "2021-12-31 23:59:59",
        "test_start": "2022-01-01 00:00:00",
        "test_end":   "2022-12-31 23:59:59",
    },
    {
        "name":       "Window 2",
        "train_end":  "2021-12-31 23:59:59",
        "val_start":  "2022-01-01 00:00:00",
        "val_end":    "2022-12-31 23:59:59",
        "test_start": "2023-01-01 00:00:00",
        "test_end":   "2023-12-31 23:59:59",
    },
    {
        "name":       "Window 3",
        "train_end":  "2022-12-31 23:59:59",
        "val_start":  "2023-01-01 00:00:00",
        "val_end":    "2023-12-31 23:59:59",
        "test_start": "2024-01-01 00:00:00",
        "test_end":   "2024-12-31 23:59:59",
    },
    {
        "name":       "Window 4",
        "train_end":  "2023-12-31 23:59:59",
        "val_start":  "2024-01-01 00:00:00",
        "val_end":    "2024-12-31 23:59:59",
        "test_start": "2025-01-01 00:00:00",
        "test_end":   "2025-12-31 23:59:59",
    },
]