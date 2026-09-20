"""
features.py
───────────
Step 5 – Create time-based and lag features for modelling.

Features produced
─────────────────
Calendar  : hour, day_of_month, month, year, weekday (0=Mon), is_weekend,
            quarter, day_of_year, week_of_year
Cyclical  : sin/cos encodings for hour and month (preserves periodicity)
Lag       : 1h, 24h, 48h, 168h (1-week) lags on Global_active_power / Energy_kWh
Rolling   : 24h rolling mean and std
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index  # DatetimeIndex

    df["hour"] = idx.hour
    df["day_of_month"] = idx.day
    df["month"] = idx.month
    df["year"] = idx.year
    df["weekday"] = idx.weekday        # 0 = Monday
    df["is_weekend"] = (idx.weekday >= 5).astype(int)
    df["quarter"] = idx.quarter
    df["day_of_year"] = idx.day_of_year
    df["week_of_year"] = idx.isocalendar().week.astype(int).values

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour (0-23) and month (1-12) as sin/cos pairs."""
    df = df.copy()
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    """Add lag columns for the target.  lags = list of periods to shift."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean and std for the target column."""
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean{w}"] = (
            df[target_col].shift(1).rolling(w).mean()
        )
        df[f"{target_col}_roll_std{w}"] = (
            df[target_col].shift(1).rolling(w).std()
        )
    return df


def build_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Full feature set for the hourly granularity (used by ML models)."""
    target = "Global_active_power"
    df = hourly.copy()
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, target, lags=[1, 24, 48, 168])
    df = add_rolling_features(df, target, windows=[24, 168])
    df.dropna(inplace=True)
    return df


def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Full feature set for the daily granularity."""
    target = "Energy_kWh"
    df = daily.copy()
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, target, lags=[1, 7, 14, 30])
    df = add_rolling_features(df, target, windows=[7, 30])
    df.dropna(inplace=True)
    return df
