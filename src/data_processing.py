"""
data_processing.py
──────────────────
Step 1 – Load the raw dataset.
Step 2 – Combine Date + Time into a proper datetime index.
Step 3 – Handle missing / invalid values, duplicates, and type coercion.
Step 4 – Aggregate 1-minute readings into hourly and daily summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("household_power_consumption.csv")

# Column names and expected numeric columns
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame with a proper DatetimeIndex."""
    df = pd.read_csv(
        path,
        sep=",",
        low_memory=False,
        na_values=["?", "NA", "", " "],
    )

    # ── Step 2: Combine Date + Time ──────────────────────────────────────────
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)

    # ── Step 3: Type coercion ─────────────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean(df: pd.DataFrame) -> dict:
    """
    Validate and clean the minute-level DataFrame.

    Returns a dict with:
        'df'         – cleaned DataFrame
        'report'     – quality report (missing counts, duplicates, outliers)
    """
    report = {}

    # Missing values
    report["missing_before"] = df.isnull().sum().to_dict()
    report["missing_pct"] = (df.isnull().mean() * 100).round(2).to_dict()

    # Duplicate timestamps
    dup_count = df.index.duplicated().sum()
    report["duplicate_timestamps"] = int(dup_count)
    df = df[~df.index.duplicated(keep="first")]

    # Forward-fill gaps up to 1 hour (60 one-minute readings), then drop
    df = df.ffill(limit=60)
    df.dropna(subset=["Global_active_power"], inplace=True)

    # Invalid physics: negative power / extreme voltage
    invalid_power = (df["Global_active_power"] < 0).sum()
    invalid_voltage = ((df["Voltage"] < 200) | (df["Voltage"] > 260)).sum()
    report["invalid_power_rows"] = int(invalid_power)
    report["invalid_voltage_rows"] = int(invalid_voltage)

    df.loc[df["Global_active_power"] < 0, "Global_active_power"] = np.nan
    df["Global_active_power"].ffill(inplace=True)

    report["missing_after"] = df.isnull().sum().to_dict()
    report["total_rows"] = len(df)
    report["date_range"] = (str(df.index.min()), str(df.index.max()))

    return {"df": df, "report": report}


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to hourly means / sums."""
    agg_funcs = {col: "mean" for col in NUMERIC_COLS if col in df.columns}
    hourly = df.resample("h").agg(agg_funcs)
    # Energy in kWh: mean power (kW) × 1 hour
    hourly["Energy_kWh"] = hourly["Global_active_power"]
    return hourly.dropna(subset=["Global_active_power"])


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to daily sums / means."""
    daily = df.resample("D").agg(
        {
            "Global_active_power": "mean",
            "Global_reactive_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum",
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
        }
    )
    # Daily energy in kWh: mean power (kW) × 24 h
    daily["Energy_kWh"] = daily["Global_active_power"] * 24
    return daily.dropna(subset=["Global_active_power"])


def run_pipeline(path: Path = RAW_PATH) -> dict:
    """End-to-end loading pipeline. Returns all granularities + QA report."""
    raw = load_raw(path)
    result = clean(raw)
    df_clean = result["df"]
    hourly = aggregate_hourly(df_clean)
    daily = aggregate_daily(df_clean)
    return {
        "minute": df_clean,
        "hourly": hourly,
        "daily": daily,
        "report": result["report"],
    }
