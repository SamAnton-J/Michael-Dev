"""
model.py
Proposed model: XGBoost gradient boosted trees with TimeSeriesSplit CV.

Why XGBoost:
  - Captures non-linear relationships between fuel prices and electricity prices
  - Robust to outliers (extreme prices during the 2021-22 energy crisis)
  - No stationarity assumption — handles structural market shifts
  - Fast training with parallelism via GridSearchCV
  - Built-in feature importance for interpretability

Training strategy:
  - StandardScaler in pipeline for consistency (trees are scale-invariant
    but included for uniformity across all models)
  - TimeSeriesSplit (3 folds) for hyperparameter tuning — respects
    temporal ordering, never uses future data to validate past
  - GridSearchCV over 32 parameter combinations × 3 folds = 96 fits per zone
  - Final model trained on full training set, evaluated on held-out test
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import TARGET, FEATURES, MODEL_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameter search grid ─────────────────────────────────────────────────
PARAM_GRID = {
    "model__n_estimators":     [300, 500],
    "model__max_depth":        [4, 6],
    "model__learning_rate":    [0.05, 0.1],
    "model__subsample":        [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}


class XGBModel:
    """
    XGBoost regressor with time-series cross-validation for hyperparameter tuning.
    Wraps GridSearchCV with TimeSeriesSplit so validation folds always follow
    training folds chronologically.
    """

    def __init__(self, n_splits: int = 3, n_jobs: int = -1):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,        # parallelism handled by GridSearchCV
                verbosity=0,
            )),
        ])
        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.search = GridSearchCV(
            pipeline,
            PARAM_GRID,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=n_jobs,
            verbose=0,
            refit=True,
        )
        self.feature_cols = []

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = [c for c in FEATURES if c in train_df.columns]
        X = train_df[self.feature_cols].values
        y = train_df[TARGET].values
        print(f"    Fitting XGBoost: {len(train_df):,} rows × "
              f"{len(self.feature_cols)} features...")
        self.search.fit(X, y)
        print(f"    Best params: {self.search.best_params_}")
        print(f"    Best CV MAE: {-self.search.best_score_:.4f}")
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].values
        return self.search.predict(X)

    def feature_importance(self) -> pd.Series:
        """
        Return XGBoost gain-based feature importances sorted descending.
        Gain measures how much each feature improves splits weighted by
        the number of samples it covers — more meaningful than frequency.
        """
        model = self.search.best_estimator_.named_steps["model"]
        return pd.Series(
            model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)


if __name__ == "__main__":
    from data_loader import load_processed, get_zone, get_splits
    from metrics import evaluate

    df    = load_processed()
    zdf   = get_zone(df, "NO1")
    train, val, test = get_splits(zdf)

    m     = XGBModel(n_splits=3)
    m.fit(train)
    preds = m.predict(test)

    result = evaluate(test[TARGET].values, preds, "xgboost", "NO1")
    print(f"\nNO1 XGBoost — MAE: {result['MAE']}  RMSE: {result['RMSE']}")
    print("\nTop 10 features:")
    print(m.feature_importance().head(10))