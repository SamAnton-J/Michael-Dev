"""
evaluate.py
Rolling window experiment runner — fits all models across all zones
using an expanding training window.

Rolling window structure (4 windows):
  Window 1: train=2020–2021, val=2022, test=2022
  Window 2: train=2020–2022, val=2023, test=2023
  Window 3: train=2020–2023, val=2024, test=2024
  Window 4: train=2020–2024, val=2025, test=2025  ← final window

For each window × zone combination:
  1. Slice train/val/test from processed dataset
  2. Fit all 5 models on train (val used for LSTM early stopping)
  3. Evaluate all 5 models on test
  4. Save metrics, feature importances and forecast plots

Run: python evaluate.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    TARGET, ZONES, FEATURES,
    ROLLING_WINDOWS, MODEL_DIR, PLOT_DIR, OUTPUT_DIR,
)
from data_loader import load_processed, get_zone, get_rolling_window_splits
from benchmarks import naive_forecast, ExpertRedAdv, ExpertMlpAdvHyper
from model import XGBModel
from lstm_model import LSTMModel
from metrics import evaluate, compare_models

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Run all models for one zone × one window ──────────────────────────────────
def run_zone_window(
    zone: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    window_name: str,
) -> tuple[list[dict], dict]:
    """
    Fit all 5 models on train, evaluate on test.
    Val is passed to LSTM for early stopping.

    Returns:
        results     — list of metric dicts (one per model)
        predictions — dict of model_name → np.ndarray of predictions
    """
    actual      = test[TARGET].values
    results     = []
    predictions = {"actual": actual}

    # 1. Naive — no training, reads price_lag24 directly
    print(f"    [naive]")
    naive_pred = naive_forecast(test)
    results.append(evaluate(actual, naive_pred, "naive", zone, window_name))
    predictions["naive"] = naive_pred

    # 2. ExpertRedAdv — Lasso on day-ahead features only
    print(f"    [expert_redadv]")
    redadv      = ExpertRedAdv().fit(train)
    redadv_pred = redadv.predict(test)
    results.append(evaluate(actual, redadv_pred, "expert_redadv", zone, window_name))
    predictions["expert_redadv"] = redadv_pred

    # 3. ExpertMlpAdvHyper — MLP with grid search
    print(f"    [expert_mlp_advhyper]")
    mlp      = ExpertMlpAdvHyper().fit(train)
    mlp_pred = mlp.predict(test)
    results.append(evaluate(actual, mlp_pred, "expert_mlp_advhyper", zone, window_name))
    predictions["expert_mlp_advhyper"] = mlp_pred

    # 4. XGBoost — proposed model with TimeSeriesSplit CV
    print(f"    [xgboost]")
    xgb      = XGBModel(n_splits=3)
    xgb.fit(train)
    xgb_pred = xgb.predict(test)
    results.append(evaluate(actual, xgb_pred, "xgboost", zone, window_name))
    predictions["xgboost"] = xgb_pred

    # Save XGBoost feature importances
    fi = xgb.feature_importance()
    fi_path = MODEL_DIR / f"{window_name}_{zone}_feature_importance.csv"
    fi.to_csv(fi_path, header=["importance"])

    # 5. LSTM — 48-hour sliding window sequence model
    print(f"    [lstm]")
    lstm      = LSTMModel(seq_len=48, epochs=50, batch_size=256)
    lstm.fit(train)
    lstm_pred = lstm.predict_with_context(train, test)
    results.append(evaluate(actual, lstm_pred, "lstm", zone, window_name))
    predictions["lstm"] = lstm_pred

    return results, predictions


# ── Forecast plot for one zone × one window ───────────────────────────────────
def plot_forecast(
    zone: str,
    test: pd.DataFrame,
    predictions: dict,
    window_name: str,
    sample_days: int = 14,
):
    """
    Plot actual vs all model forecasts for the first sample_days of the test set.
    14 days = 2 full weekly cycles — enough to show daily and weekly patterns.
    """
    idx    = pd.to_datetime(test["time_utc"])
    actual = predictions["actual"]
    n      = sample_days * 24

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(idx[:n], actual[:n], label="Actual", color="black",
            linewidth=1.2, zorder=5)

    colours = {
        "naive":               "grey",
        "expert_redadv":       "blue",
        "expert_mlp_advhyper": "green",
        "xgboost":             "red",
        "lstm":                "purple",
    }
    for model_name, preds in predictions.items():
        if model_name == "actual":
            continue
        ax.plot(
            idx[:n], preds[:n],
            label=model_name,
            alpha=0.7,
            linewidth=0.9,
            color=colours.get(model_name, "orange"),
        )

    ax.set_title(f"{zone} — {window_name}: Forecast vs Actual "
                 f"(first {sample_days} days of test set)")
    ax.set_ylabel("EUR/MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend()
    fig.tight_layout()

    path = PLOT_DIR / f"{zone}_{window_name}_forecast.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"    Plot saved → {path.name}")


# ── Main rolling window loop ───────────────────────────────────────────────────
def run_all():
    print("Loading processed dataset...")
    df = load_processed()
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    all_results = []

    for window in ROLLING_WINDOWS:
        wname = window["name"]
        print("=" * 60)
        print(f"{wname}")
        print("=" * 60)

        for zone in ZONES:
            print(f"\n  Zone: {zone}")

            zdf = get_zone(df, zone)
            train, val, test = get_rolling_window_splits(
                zdf,
                train_end  = window["train_end"],
                val_start  = window["val_start"],
                val_end    = window["val_end"],
                test_start = window["test_start"],
                test_end   = window["test_end"],
            )

            print(f"    train={len(train):,}  val={len(val):,}  "
                  f"test={len(test):,}")

            if len(train) == 0 or len(test) == 0:
                print(f"    Skipping — insufficient data")
                continue

            results, predictions = run_zone_window(
                zone, train, val, test, wname
            )
            all_results.extend(results)

            # Save per-zone per-window results
            table = compare_models(results)
            table_path = MODEL_DIR / f"{wname}_{zone}_results.csv"
            table.to_csv(table_path)
            print(f"\n    Results:")
            print(table[["MAE", "RMSE", "sMAPE"]].to_string())

            # Save forecast plot
            plot_forecast(zone, test, predictions, wname)

    # Save combined results across all windows and zones
    combined = pd.DataFrame(all_results)
    combined_path = MODEL_DIR / "all_windows_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\n\nAll results saved → {combined_path}")

    # Print MAE summary table — models vs windows
    print("\n=== MAE Summary (averaged across all zones) ===")
    summary = (
        combined
        .groupby(["window", "model"])["MAE"]
        .mean()
        .round(2)
        .unstack("model")
    )
    print(summary.to_string())

    return combined


if __name__ == "__main__":
    run_all()