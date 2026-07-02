"""
config.py
Central configuration for the Norwegian electricity price forecasting project.
All constants, paths, column names and window definitions live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
RAW_CSV       = ROOT / "NO1_NO5_2020_2025_raw_data.csv"
OUTPUT_DIR    = ROOT / "outputs"
MODEL_DIR     = OUTPUT_DIR / "models"
PLOT_DIR      = OUTPUT_DIR / "plots"
PROCESSED_CSV = OUTPUT_DIR / "processed_dataset.csv"

# ── Zones ──────────────────────────────────────────────────────────────────────
ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# ── Target column ──────────────────────────────────────────────────────────────
TARGET = "Price"

# ── Feature columns (what the model receives as input) ─────────────────────────
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_winter",
    "price_lag24", "price_lag168",
    "Load_DayAhead", "load_actual_lag1",
    "WindOnshore_DayAhead", "wind_actual_lag1",
    "gas_lag1d", "coal_lag1d", "oil_lag1d", "eua_lag1d",
]

# ── Main train / val / test split ──────────────────────────────────────────────
# Full end-of-day timestamps ensure the complete last day of each period
# is captured. Without the time component, pandas treats "2023-12-31" as
# "2023-12-31 00:00:00", missing all 23 hours after midnight.
TRAIN_END  = "2023-12-31 23:59:59"
VAL_START  = "2024-01-01 00:00:00"
VAL_END    = "2024-12-31 23:59:59"
TEST_START = "2025-01-01 00:00:00"
TEST_END   = "2025-12-31 23:59:59"

# ── Rolling window definitions ─────────────────────────────────────────────────
# Expanding training window — each window adds one more year to training.
# Training always starts from 2020-01-08 (after warmup drop).
# Val and test are always the next two consecutive years.
ROLLING_WINDOWS = [
    {
        "name":       "Window 1",
        "train_end":  "2021-12-31 23:59:59",
        "val_start":  "2022-01-01 00:00:00",
        "val_end":    "2022-12-31 23:59:59",
        "test_start": "2022-01-01 00:00:00",
        "test_end":   "2022-12-31 23:59:59",
    },
    {
        "name":       "Window 2",
        "train_end":  "2022-12-31 23:59:59",
        "val_start":  "2023-01-01 00:00:00",
        "val_end":    "2023-12-31 23:59:59",
        "test_start": "2023-01-01 00:00:00",
        "test_end":   "2023-12-31 23:59:59",
    },
    {
        "name":       "Window 3",
        "train_end":  "2023-12-31 23:59:59",
        "val_start":  "2024-01-01 00:00:00",
        "val_end":    "2024-12-31 23:59:59",
        "test_start": "2024-01-01 00:00:00",
        "test_end":   "2024-12-31 23:59:59",
    },
    {
        "name":       "Window 4",
        "train_end":  "2024-12-31 23:59:59",
        "val_start":  "2025-01-01 00:00:00",
        "val_end":    "2025-12-31 23:59:59",
        "test_start": "2025-01-01 00:00:00",
        "test_end":   "2025-12-31 23:59:59",
    },
]

# ── Timezone ───────────────────────────────────────────────────────────────────
TIMEZONE = "Europe/Oslo"   # CET (UTC+1 winter) / CEST (UTC+2 summer)

# ── Columns to drop from raw data ──────────────────────────────────────────────
COLS_DROP = [
    "Generation_Solar_Actual",
    "Generation_Solar_DayAhead",
    "Generation_WindOffshore_Actual",
    "Generation_WindOffshore_DayAhead",
    "date",
]

# ── USD fuel columns (need EUR conversion) ────────────────────────────────────
COLS_USD_FUEL = ["gas_fM_01", "coal_fM_01", "oil_fM_01"]
COL_EUR_FUEL  = "EUA_fM_01"
COL_FX        = "USD_EUR"

# ── Columns that need forward-fill (weekend/holiday gaps) ────────────────────
COLS_FFILL_FUEL = ["gas_fM_01", "coal_fM_01", "oil_fM_01", "EUA_fM_01", "USD_EUR"]

# ── Load and wind columns ─────────────────────────────────────────────────────
COLS_LOAD = ["Load_Actual", "Load_DayAhead"]
COLS_WIND = ["Generation_WindOnshore_Actual", "Generation_WindOnshore_DayAhead"]

# ── Renamed wind columns (shorter names in processed CSV) ────────────────────
WIND_RENAME = {
    "Generation_WindOnshore_Actual":   "WindOnshore_Actual",
    "Generation_WindOnshore_DayAhead": "WindOnshore_DayAhead",
}

# ── Zones with no wind data ───────────────────────────────────────────────────
ZONES_NO_WIND = ["NO5"]

# ── Forward-fill limits ───────────────────────────────────────────────────────
# Wind DayAhead has gaps up to 48h, Actual up to 22h — use 48 to cover all
# Load has only isolated 1-2h gaps — limit=3 is sufficient
FFILL_LIMIT_WIND = 48
FFILL_LIMIT_LOAD = 3

# ── Warmup rows to drop after lagging ────────────────────────────────────────
# price_lag168 needs 168 rows of history (7 days) before it is valid.
# Dropping the first 168 rows per zone removes Jan 1-7 2020 — <0.3% of data.
WARMUP_ROWS = 168