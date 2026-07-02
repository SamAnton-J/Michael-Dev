"""
lstm_model.py
LSTM deep learning model for electricity price forecasting.

Architecture:
  Input: 48-hour sliding window of all features
  LSTM(64, return_sequences=True) → Dropout(0.2)
  LSTM(32)                        → Dropout(0.2)
  Dense(1)                        → price prediction

Why LSTM over a simple MLP:
  - Receives a sequence of 48 hours, not a single snapshot
  - Gated memory cells capture multi-hour price momentum
  - Forget gate prevents vanishing gradients over long sequences
  - Can learn that "prices rising for 6 hours" is a different signal
    from "prices at the same level for 6 hours"

Key design:
  - predict_with_context() prepends last 48 train rows to test set
    so the very first test hour gets a full context window
  - StandardScaler applied to both features and target
  - Target is inverse-transformed after prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from config import TARGET, FEATURES

SEQUENCE_LENGTH = 48   # 48-hour context window


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Convert flat feature matrix into 3D sequences for LSTM input.

    Input:  X shape (n_samples, n_features)
            y shape (n_samples,)
    Output: Xs shape (n_samples - seq_len, seq_len, n_features)
            ys shape (n_samples - seq_len,)

    Each Xs[i] is the window X[i : i+seq_len].
    ys[i] is the target at position i+seq_len.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_lstm(n_features: int, seq_len: int) -> tf.keras.Model:
    """
    Build a stacked LSTM using the Keras functional API.
    Using functional API avoids the input_shape deprecation warning
    that appears with Sequential + input_shape argument in Keras 3.x.
    """
    inputs = Input(shape=(seq_len, n_features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mae",
    )
    return model


class LSTMModel:
    """
    LSTM wrapper with the same fit/predict interface as XGBModel.
    Handles its own scaling, sequence construction and training callbacks.
    """

    def __init__(
        self,
        seq_len: int = SEQUENCE_LENGTH,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler_X   = StandardScaler()
        self.scaler_y   = StandardScaler()
        self.model      = None
        self.feature_cols = []

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = [c for c in FEATURES if c in train_df.columns]

        X = train_df[self.feature_cols].values
        y = train_df[TARGET].values.reshape(-1, 1)

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y).ravel()

        Xs, ys = make_sequences(X_scaled, y_scaled, self.seq_len)

        # 90% of sequences for training, 10% for internal validation
        split  = int(len(Xs) * 0.9)
        X_tr, X_val = Xs[:split], Xs[split:]
        y_tr, y_val = ys[:split], ys[split:]

        self.model = build_lstm(
            n_features=Xs.shape[2],
            seq_len=self.seq_len,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        print(f"    Fitting LSTM: {len(X_tr):,} sequences "
              f"(seq_len={self.seq_len}, features={Xs.shape[2]})...")

        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_with_context(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict on test set using the last seq_len rows of training data
        as context for the first test window.

        Without this, the first seq_len test hours would have no valid
        context window and would be NaN.
        """
        context = train_df[self.feature_cols].iloc[-self.seq_len:]
        full_X  = pd.concat([context, test_df[self.feature_cols]])
        y_dummy = np.zeros(len(full_X))

        X_scaled = self.scaler_X.transform(full_X.values)
        Xs, _    = make_sequences(X_scaled, y_dummy, self.seq_len)

        preds_scaled = self.model.predict(Xs, verbose=0).ravel()
        preds = self.scaler_y.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()
        return preds

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict on test set without external context.
        First seq_len predictions will be NaN.
        Use predict_with_context() for accurate evaluation.
        """
        X = test_df[self.feature_cols].values
        y_dummy  = np.zeros(len(X))
        X_scaled = self.scaler_X.transform(X)
        Xs, _    = make_sequences(X_scaled, y_dummy, self.seq_len)

        preds_scaled = self.model.predict(Xs, verbose=0).ravel()
        preds = self.scaler_y.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()

        full_preds = np.full(len(test_df), np.nan)
        full_preds[self.seq_len:] = preds
        return full_preds


if __name__ == "__main__":
    from data_loader import load_processed, get_zone, get_splits
    from metrics import evaluate

    df    = load_processed()
    zdf   = get_zone(df, "NO1")
    train, val, test = get_splits(zdf)

    m     = LSTMModel(seq_len=48, epochs=50, batch_size=256)
    m.fit(train)
    preds = m.predict_with_context(train, test)

    result = evaluate(test[TARGET].values, preds, "lstm", "NO1")
    print(f"\nNO1 LSTM — MAE: {result['MAE']}  RMSE: {result['RMSE']}")