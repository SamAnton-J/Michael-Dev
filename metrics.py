"""
metrics.py
Forecast evaluation metrics for electricity price forecasting.

Four standard metrics:
  MAE   — mean absolute error (EUR/MWh) — primary ranking metric
  RMSE  — root mean squared error — penalises large spikes more
  MAPE  — mean absolute percentage error — skips near-zero prices
  sMAPE — symmetric MAPE — more stable near zero than MAPE
"""

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error in EUR/MWh."""
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error — penalises large errors more than MAE."""
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mape(actual: np.ndarray, forecast: np.ndarray, zero_threshold: float = 1.0) -> float:
    """
    Mean absolute percentage error.
    Skips hours where |actual| <= zero_threshold to avoid division by near-zero.
    zero_threshold=1.0 EUR/MWh is appropriate for Norwegian electricity prices.
    """
    mask = np.abs(actual) > zero_threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Symmetric MAPE — uses average of actual and forecast as denominator.
    More stable than MAPE near zero prices (NO3, NO4 zones).
    """
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    mask  = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100)


def evaluate(
    actual: np.ndarray,
    forecast: np.ndarray,
    model_name: str = "",
    zone: str = "",
    window: str = "",
) -> dict:
    """
    Compute all four metrics and return as a dict.
    Optionally includes model name, zone and window for result tracking.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)

    # Remove NaN pairs (e.g. LSTM warmup period)
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]

    return {
        "model":  model_name,
        "zone":   zone,
        "window": window,
        "MAE":    round(mae(a, f), 4),
        "RMSE":   round(rmse(a, f), 4),
        "MAPE":   round(mape(a, f), 4),
        "sMAPE":  round(smape(a, f), 4),
        "n":      len(a),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of evaluate() dicts to a DataFrame sorted by MAE.
    """
    df = pd.DataFrame(results)
    if "model" in df.columns:
        df = df.set_index("model")
    return df.sort_values("MAE")


if __name__ == "__main__":
    # Quick sanity check
    actual   = np.array([10.0, 20.0, 30.0, 0.5, -5.0])
    forecast = np.array([11.0, 19.0, 32.0, 0.6, -4.5])

    result = evaluate(actual, forecast, model_name="test", zone="NO1", window="W1")
    print("Test evaluation:")
    for k, v in result.items():
        print(f"  {k}: {v}")