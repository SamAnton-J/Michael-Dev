# Electricity Price Forecasting — Norwegian Zones (Group D)

Day-ahead electricity price forecasting for the five Norwegian bidding zones **NO1–NO5** (2020–2025).

This project implements a complete forecasting pipeline: data preprocessing, feature engineering, rolling window evaluation across four time windows, five models, and PDF report generation. Built for **Group D (3P)**.

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| `naive` | Baseline | Price(t) = Price(t−24) — same hour yesterday. No training. |
| `expert_redadv` | ML | Lasso regression on day-ahead features only (no fuel lags, no real-time lags) |
| `expert_mlp_advhyper` | ML | MLP with GridSearchCV over layer sizes, regularisation and learning rate |
| `xgboost` | ML | **Proposed model** — gradient boosted trees with TimeSeriesSplit CV |
| `lstm` | DL | Stacked LSTM with 48-hour sliding window input |

---

## Rolling Window Evaluation

Instead of a single train/test split, models are evaluated across four expanding windows. Each window adds one year of training data and tests on the following year:

| Window | Train | Val | Test |
|--------|-------|-----|------|
| Window 1 | 2020 | 2022 | 2022 |
| Window 2 | 2020–2021 | 2023 | 2023 |
| Window 3 | 2020–2022 | 2024 | 2024 |
| Window 4 | 2020–2023 | 2025 | 2025 |

This reveals how performance changes as training data grows, and exposes how models behave during structural market regime changes (e.g. the 2021–2022 energy crisis).

**Val** is used only for LSTM early stopping and MLP hyperparameter tuning — never for final metric reporting.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SamAnton-J/Michael-Dev.git
cd Michael-Dev
```

### 2. Create virtual environment and install dependencies

```bash
python3.11 -m venv doc
source doc/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

### 3. Place the raw dataset

Copy `NO1_NO5_2020_2025_raw_data.csv` into the project root.

The CSV is ~33 MB and gitignored. `config.py` points to:

```
ROOT / "NO1_NO5_2020_2025_raw_data.csv"
```

where `ROOT` is the project folder — no absolute path changes needed as long as the CSV is in the same folder as the scripts.

### 4. Run the pipeline

```bash
# Step 1 — preprocess raw data (~30 seconds)
python preprocess.py

# Step 2 — run all models across all windows (~2–3 hours)
python evaluate.py
```

---

## Project Structure

```
Michael-Dev/
├── config.py              # All constants: paths, zones, features, split dates, rolling windows
├── preprocess.py          # Full 14-step preprocessing pipeline → outputs/processed_dataset.csv
├── data_loader.py         # Loads processed CSV, returns per-zone train/val/test splits
├── metrics.py             # MAE, RMSE, MAPE, sMAPE evaluation functions
├── benchmarks.py          # Naive, ExpertRedAdv (Lasso), ExpertMlpAdvHyper (MLP)
├── model.py               # Proposed XGBoost model with TimeSeriesSplit CV
├── lstm_model.py          # Stacked LSTM with 48-hour sliding window
├── evaluate.py            # Rolling window experiment runner — all models, all zones
├── requirements.txt
├── Readme.md
├── .gitignore
│
├── NO1_NO5_2020_2025_raw_data.csv   # Raw data (gitignored)
├── doc/                              # Virtual environment (gitignored)
│
└── outputs/                          # All generated files (gitignored)
    ├── processed_dataset.csv         # Clean feature dataset from preprocess.py
    ├── models/
    │   ├── Window 1_NO1_results.csv  # Metrics per window per zone
    │   ├── Window 1_NO1_feature_importance.csv
    │   ├── ...                       # Same pattern for all windows and zones
    │   └── all_windows_results.csv   # Combined results across all windows and zones
    └── plots/
        ├── NO1_Window 1_forecast.png # 14-day forecast plot per zone per window
        └── ...                       # Same pattern for all windows and zones
```

---

## Source Modules

### `config.py`

Single source of truth for all constants. Edit here — changes propagate everywhere.

| Constant | Value | Purpose |
|----------|-------|---------|
| `ROOT` | `Path(__file__).parent` | Project root — all paths relative to this |
| `RAW_CSV` | `ROOT / "NO1_NO5_2020_2025_raw_data.csv"` | Raw input file |
| `PROCESSED_CSV` | `outputs/processed_dataset.csv` | Output of preprocess.py |
| `ZONES` | `["NO1"..."NO5"]` | All five Norwegian bidding zones |
| `TARGET` | `"Price"` | Target column name |
| `FEATURES` | List of 15 feature names | Features fed to every model |
| `TRAIN_END` | `"2023-12-31 23:59:59"` | End of main training period |
| `VAL_START/END` | `"2024-*"` | Main validation period |
| `TEST_START/END` | `"2025-*"` | Main test period |
| `ROLLING_WINDOWS` | List of 4 window dicts | Rolling window boundaries |
| `TIMEZONE` | `"Europe/Oslo"` | CET/CEST for time feature extraction |
| `FFILL_LIMIT_WIND` | `48` | Max hours to forward-fill wind gaps |
| `FFILL_LIMIT_LOAD` | `3` | Max hours to forward-fill load gaps |
| `WARMUP_ROWS` | `168` | Rows dropped per zone after lagging (7 days) |

---

### `preprocess.py`

Transforms the raw 366,952-row CSV into a clean 256,589-row feature dataset. Runs 14 steps in sequence — each step has a single responsibility.

| Step | What it does | Why |
|------|-------------|-----|
| 1 | Filter to hourly rows (`minute == 0`) | Dataset switches to 15-min resolution from Mar 2025 — sub-hourly rows corrupt lag features |
| 2 | Drop solar, offshore wind, date columns | 96–100% null — no solar capacity in Norway, offshore wind negligible |
| 3 | Forward-fill fuels and USD_EUR per zone | Fuel/FX markets closed weekends — structured gaps, not random missingness |
| 4 | Convert USD fuels to EUR (`gas × USD_EUR`) | All features and target must be in same currency |
| 5 | Impute wind and load nulls | Load: ffill limit=3. Wind NO1-NO4: ffill limit=48. Wind NO5: fill 0 (no wind data) |
| 6 | Drop rows where Price is null | Target cannot be imputed — would corrupt training labels |
| 7 | Convert UTC → CET (`Europe/Oslo`) | Nord Pool operates in CET — time features extracted from CET give consistent peak hours |
| 8 | Create time features from CET | Plain integers: hour, day_of_week, month, is_weekend, is_winter |
| 9 | Create lag features per zone | price_lag24, price_lag168, load_actual_lag1, wind_actual_lag1, fuel_lag1d × 4 |
| 10 | Drop raw columns replaced by lags | Prevents model from accidentally using unlagged (future) values |
| 11 | Drop first 168 warmup rows per zone | price_lag168 needs 168 rows of history before it is valid |
| 12 | Add split column | train=2020–2023, val=2024, test=2025 |
| 13 | Assert zero nulls | Hard error if any nulls remain — final CSV must be completely clean |
| 14 | Reorder and save | 20 columns in logical order, Price always last |

---

### `data_loader.py`

Reads `processed_dataset.csv` and slices it into train/val/test DataFrames.

| Function | Returns | Purpose |
|----------|---------|---------|
| `load_processed()` | Full DataFrame | Load CSV with time_utc parsed as datetime |
| `get_zone(df, zone)` | Zone DataFrame | Filter to one zone, sorted by time_utc |
| `get_splits(zone_df)` | (train, val, test) | Split using pre-computed split column |
| `get_X_y(df)` | (X, y, feature_cols) | Extract feature matrix and target vector |
| `get_rolling_window_splits(...)` | (train, val, test) | Slice for a specific rolling window boundary |

---

### `metrics.py`

Four evaluation metrics. All handle NaN pairs (LSTM warmup) automatically.

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | `mean(|actual - forecast|)` | Primary ranking metric. EUR/MWh. Easy to interpret. |
| RMSE | `sqrt(mean((actual - forecast)²))` | Penalises large spike errors more than MAE |
| MAPE | `mean(|error/actual|) × 100` | Skips hours where `|actual| <= 1.0` EUR/MWh |
| sMAPE | `mean(|error| / avg(|actual|,|forecast|)) × 100` | More stable near zero — preferred for NO3/NO4 |

---

### `benchmarks.py`

**Naive** — reads `price_lag24` directly. No training. Hardest to beat in electricity forecasting because 24h autocorrelation is very strong.

**ExpertRedAdv** — Lasso regression (α=0.1) using only features available before the day-ahead auction closes (d-1 08:00). Excludes `_lag1` and `_lag1d` features. Tests how much value the real-time and fuel information adds.

**ExpertMlpAdvHyper** — MLP with 3-fold GridSearchCV over:
- Hidden layer sizes: `(64,32)`, `(128,64)`, `(64,64,32)`
- L2 regularisation alpha: `0.001`, `0.01`, `0.1`
- Learning rate: `0.001`, `0.01`

---

### `model.py`

XGBoost gradient boosted trees. TimeSeriesSplit CV (3 folds) ensures validation folds always follow training folds chronologically.

Hyperparameter grid (32 combinations × 3 folds = 96 fits per zone per window):

| Parameter | Values | Effect |
|-----------|--------|--------|
| `n_estimators` | 300, 500 | Number of trees |
| `max_depth` | 4, 6 | Tree depth — controls overfitting |
| `learning_rate` | 0.05, 0.1 | Shrinkage per step |
| `subsample` | 0.8, 1.0 | Row sampling per tree |
| `colsample_bytree` | 0.8, 1.0 | Feature sampling per tree |

Saves gain-based feature importances to `outputs/models/{window}_{zone}_feature_importance.csv`.

---

### `lstm_model.py`

Stacked LSTM with 48-hour sliding window input.

| Component | Detail |
|-----------|--------|
| Input | 48-hour window × 15 features = 3D tensor (batch, 48, 15) |
| Architecture | `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)` |
| Optimizer | Adam (lr=0.001) |
| Loss | MAE |
| Early stopping | patience=5, restores best weights |
| LR reduction | ReduceLROnPlateau, factor=0.5, patience=3, min_lr=1e-5 |
| Scaling | StandardScaler on features and target — inverse-transformed after prediction |
| Context | `predict_with_context()` prepends last 48 train rows so first test hour has a full window |

---

### `evaluate.py`

Main experiment runner. Loops over 4 windows × 5 zones × 5 models = **100 model fits**.

For each window × zone:
1. Slice train/val/test using rolling window boundaries
2. Fit all 5 models on train
3. Evaluate all 5 models on test
4. Save results CSV, feature importance CSV, forecast plot

After all windows complete, saves `all_windows_results.csv` — the master results table.

---

## Data

### Raw dataset

| Property | Value |
|----------|-------|
| File | `NO1_NO5_2020_2025_raw_data.csv` |
| Rows | 366,952 (including sub-hourly rows) |
| Columns | 17 |
| Granularity | Hourly (UTC), switches to 15-min from Mar 2025 |
| Sources | ENTSO-E Transparency Platform, Refinitiv Datastream |

### Processed dataset

| Property | Value |
|----------|-------|
| File | `outputs/processed_dataset.csv` |
| Rows | 256,589 |
| Columns | 20 |
| Nulls | Zero |
| Currencies | All EUR |
| Time features | Extracted from CET (not UTC) |

### Processed dataset columns

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `time_utc` | datetime | Original UTC timestamp |
| 2 | `time_cet` | datetime | CET/CEST local timestamp (UTC+1 winter, UTC+2 summer) |
| 3 | `zone` | string | Bidding zone: NO1–NO5 |
| 4 | `split` | string | train / val / test |
| 5 | `hour` | int | Hour of day 0–23 (from CET) |
| 6 | `day_of_week` | int | 0=Monday to 6=Sunday (from CET) |
| 7 | `month` | int | 1–12 (from CET) |
| 8 | `is_weekend` | int | 1 if Saturday or Sunday |
| 9 | `is_winter` | int | 1 if December, January or February |
| 10 | `price_lag24` | float | Price 24 hours ago (EUR/MWh) |
| 11 | `price_lag168` | float | Price 168 hours ago / same hour last week (EUR/MWh) |
| 12 | `Load_DayAhead` | float | Day-ahead load forecast (MW) — published d-1 at 08:00 |
| 13 | `load_actual_lag1` | float | Actual load 1 hour ago (MW) |
| 14 | `WindOnshore_DayAhead` | float | Day-ahead wind forecast (MW) — published d-1 at 08:00 |
| 15 | `wind_actual_lag1` | float | Actual wind generation 1 hour ago (MW) |
| 16 | `gas_lag1d` | float | Gas price previous day (EUR/MMBtu) |
| 17 | `coal_lag1d` | float | Coal price previous day (EUR/tonne) |
| 18 | `oil_lag1d` | float | Oil price previous day (EUR/bbl) |
| 19 | `eua_lag1d` | float | EU carbon allowance price previous day (EUR/tonne) |
| 20 | `Price` | float | **TARGET** — day-ahead electricity price (EUR/MWh) |

### Zone characteristics

| Zone | Median price | Cluster | Notes |
|------|-------------|---------|-------|
| NO1 | 54 EUR/MWh | Southern | Strong interconnection to Denmark, Germany |
| NO2 | 60 EUR/MWh | Southern | Highest prices — most exposed to continental gas market |
| NO3 | 20 EUR/MWh | Northern | Hydro-dominated, low prices, limited export capacity |
| NO4 | 15 EUR/MWh | Northern | Cheapest zone — pure hydro, very limited transmission south |
| NO5 | 51 EUR/MWh | Southern | Similar to NO1, no wind data reported |

---

## Outputs

### `outputs/processed_dataset.csv`
Clean preprocessed feature dataset. 256,589 rows × 20 columns. Zero nulls. All EUR. Generated by `preprocess.py`.

### `outputs/models/Window N_ZONEx_results.csv`
Metrics table for one window × one zone. Contains MAE, RMSE, MAPE, sMAPE for all 5 models sorted by MAE. 20 files total (4 windows × 5 zones).

### `outputs/models/Window N_ZONEx_feature_importance.csv`
XGBoost gain-based feature importances for one window × one zone. Shows which features drove XGBoost predictions. 20 files total.

### `outputs/models/all_windows_results.csv`
Master results file. Every model × every zone × every window in one flat CSV. Use this for all analysis and report tables.

```python
import pandas as pd
df = pd.read_csv("outputs/models/all_windows_results.csv")

# Average MAE by model and window
pivot = df.groupby(["window","model"])["MAE"].mean().round(2).unstack("model")
print(pivot)
```

### `outputs/plots/ZONEx_Window N_forecast.png`
14-day forecast plot for one zone × one window. Shows actual price (black) vs all five model predictions (grey/blue/green/red/purple) for the first two weeks of the test period. 20 plots total.

---

## Key Design Decisions

**Why forward-fill for fuels but not Price?**
Fuel nulls are structured weekend gaps — the last known price is the correct economic signal. Price nulls are random market reporting failures — the target cannot be imputed without corrupting training labels.

**Why CET for time features?**
Nord Pool clears prices in CET. Morning demand peaks at 08:00–09:00 CET. Extracting hour from UTC would shift the peak by 1–2 hours depending on DST, confusing the model.

**Why convert USD fuels to EUR?**
All features and the target are now in the same currency. The model does not need to implicitly learn the EUR/USD relationship — it is baked in at preprocessing time.

**Why drop 15-minute rows?**
ENTSO-E switched reporting format from hourly to 15-minute in March 2025. Sub-hourly rows carry no day-ahead price and corrupt `shift(24)` lag calculations because the shift operates on row position not time.

**Why TimeSeriesSplit not k-fold for XGBoost CV?**
k-fold randomly places future hours in training folds — data leakage. TimeSeriesSplit always validates on data that comes after the training fold, mimicking real deployment.

**Why 168-row warmup drop?**
`price_lag168` requires 168 rows (7 days) of history before the first valid value exists. Rows 1–168 of each zone have NaN in this column. Dropping them removes Jan 1–7 2020 per zone — less than 0.3% of data.

---

## Gitignored Files

| Path | Reason |
|------|--------|
| `doc/` | Python virtual environment (~374 MB) |
| `outputs/` | Regenerated by scripts |
| `NO1_NO5_2020_2025_raw_data.csv` | Large raw dataset (~33 MB) |
| `__pycache__/`, `.DS_Store` | Runtime / OS artifacts |

---

## License and Attribution

Data: ENTSO-E Transparency Platform (transparency.entsoe.eu) and Refinitiv Datastream.
Assignment: Group D (3P) — Norwegian electricity price forecasting, zones NO1–NO5.