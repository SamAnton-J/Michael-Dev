"""
data_loader.py
Loads the processed dataset and returns train/val/test splits per zone.

The processed CSV is produced by preprocess.py and is already clean:
  - Zero nulls
  - All currencies in EUR
  - CET time features
  - Lag features computed and causally correct
  - Split column labelled (train / val / test)

This module simply reads that CSV and slices it into the right pieces
for training and evaluation.
"""

import pandas as pd
from config import PROCESSED_CSV, ZONES, FEATURES, TARGET


def load_processed() -> pd.DataFrame:
    """
    Load the full processed dataset from CSV.
    Parses time_utc as datetime.
    """
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["time_utc"])
    return df


def get_zone(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    """
    Filter the full dataset to a single zone.
    Returns rows sorted by time_utc.
    """
    return df[df["zone"] == zone].sort_values("time_utc").reset_index(drop=True)


def get_splits(zone_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single-zone DataFrame into train, val and test sets
    using the pre-computed split column.

    Returns:
        train  — 2020–2023
        val    — 2024
        test   — 2025
    """
    train = zone_df[zone_df["split"] == "train"].reset_index(drop=True)
    val   = zone_df[zone_df["split"] == "val"].reset_index(drop=True)
    test  = zone_df[zone_df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def get_X_y(df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix X and target vector y from a DataFrame.
    Only uses columns defined in FEATURES (from config.py).
    Drops any feature columns not present (e.g. wind missing for NO5).
    """
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].values
    y = df[TARGET].values
    return X, y, available_features


def get_rolling_window_splits(
    zone_df: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Slice a zone DataFrame into train/val/test for a specific rolling window.
    Training always starts from the beginning of the dataset.

    Args:
        zone_df    : single-zone DataFrame sorted by time_utc
        train_end  : last timestamp of training period
        val_start  : first timestamp of validation period
        val_end    : last timestamp of validation period
        test_start : first timestamp of test period
        test_end   : last timestamp of test period

    Returns:
        train, val, test DataFrames
    """
    train = zone_df[zone_df["time_utc"] <= train_end].reset_index(drop=True)
    val   = zone_df[
        (zone_df["time_utc"] >= val_start) &
        (zone_df["time_utc"] <= val_end)
    ].reset_index(drop=True)
    test  = zone_df[
        (zone_df["time_utc"] >= test_start) &
        (zone_df["time_utc"] <= test_end)
    ].reset_index(drop=True)
    return train, val, test


if __name__ == "__main__":
    df = load_processed()
    print(f"Full dataset: {df.shape}")
    print(f"Zones: {df['zone'].unique().tolist()}")
    print()

    for zone in ZONES:
        zdf = get_zone(df, zone)
        train, val, test = get_splits(zdf)
        X_train, y_train, feats = get_X_y(train)
        print(f"{zone}: train={len(train):,}  val={len(val):,}  "
              f"test={len(test):,}  features={len(feats)}")