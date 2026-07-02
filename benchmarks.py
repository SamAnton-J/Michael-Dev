"""
benchmarks.py
Three benchmark models for comparison against the proposed XGBoost and LSTM.

  1. Naive          — price(t) = price(t-24), no training required
  2. ExpertRedAdv   — Lasso regression on day-ahead features only
  3. ExpertMlpAdvHyper — MLP with grid-searched hyperparameters
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import TARGET, FEATURES


# ── 1. Naive ──────────────────────────────────────────────────────────────────
def naive_forecast(test_df: pd.DataFrame) -> np.ndarray:
    """
    Naive benchmark: price(t) = price(t-24).
    Reads the pre-computed price_lag24 column directly.
    No training required — just reads yesterday's same-hour price.
    This is the hardest baseline to beat in electricity forecasting
    because daily price cycles are strong and consistent.
    """
    return test_df["price_lag24"].values


# ── 2. ExpertRedAdv (Regularised Day-Ahead) ──────────────────────────────────
def _get_da_features(df: pd.DataFrame) -> list[str]:
    """
    Select only features available before the day-ahead auction (d-1 08:00).
    Excludes real-time lag features (_lag1) and fuel lag features (_lag1d)
    since those represent information the auction has not yet seen.
    Keeps: time features, price lags, Load_DayAhead, WindOnshore_DayAhead.
    """
    exclude = ["_lag1", "_lag1d"]
    return [
        c for c in FEATURES
        if c in df.columns
        and not any(e in c for e in exclude)
    ]


class ExpertRedAdv:
    """
    Lasso regression using only day-ahead available features.
    Lasso (L1 penalty) drives irrelevant feature weights to exactly zero,
    producing a sparse model appropriate when several DA features are correlated.
    """

    def __init__(self, alpha: float = 0.1):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=alpha, max_iter=5000)),
        ])
        self.feature_cols = []

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = _get_da_features(train_df)
        X = train_df[self.feature_cols].values
        y = train_df[TARGET].values
        self.pipeline.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].values
        return self.pipeline.predict(X)


# ── 3. ExpertMlpAdvHyper (MLP with grid search) ───────────────────────────────
class ExpertMlpAdvHyper:
    """
    Multi-layer perceptron with GridSearchCV over architecture and regularisation.
    Uses the full feature set (including fuel lags and real-time lags).
    'AdvHyper' = advanced hyperparameter tuning via grid search.
    """

    PARAM_GRID = {
        "model__hidden_layer_sizes": [(64, 32), (128, 64), (64, 64, 32)],
        "model__alpha":              [0.001, 0.01, 0.1],
        "model__learning_rate_init": [0.001, 0.01],
    }

    def __init__(self, cv: int = 3, n_jobs: int = -1):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )),
        ])
        self.search = GridSearchCV(
            pipeline,
            self.PARAM_GRID,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
            verbose=0,
        )
        self.feature_cols = []

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = [c for c in FEATURES if c in train_df.columns]
        X = train_df[self.feature_cols].values
        y = train_df[TARGET].values
        self.search.fit(X, y)
        print(f"    Best MLP params: {self.search.best_params_}")
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].values
        return self.search.predict(X)


if __name__ == "__main__":
    from data_loader import load_processed, get_zone, get_splits, get_X_y
    from metrics import evaluate, compare_models

    df   = load_processed()
    zdf  = get_zone(df, "NO1")
    train, val, test = get_splits(zdf)

    actual  = test[TARGET].values
    results = []

    # Naive
    naive_pred = naive_forecast(test)
    results.append(evaluate(actual, naive_pred, "naive", "NO1"))

    # ExpertRedAdv
    redadv = ExpertRedAdv().fit(train)
    results.append(evaluate(actual, redadv.predict(test), "expert_redadv", "NO1"))

    print(compare_models(results)[["MAE", "RMSE", "sMAPE"]])