"""
eda.py
──────
Step 6 – Exploratory Data Analysis.

Produces and saves static matplotlib/seaborn plots to outputs/plots/.
Every public function returns the Figure so the Streamlit dashboard
can embed it directly without re-running the analysis.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", palette="tab10")


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=150)
    return fig


# ── 1. Daily consumption trend ───────────────────────────────────────────────

def plot_daily_trend(daily: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily.index, daily["Energy_kWh"], color=PALETTE[0], lw=0.8, alpha=0.7)
    # 30-day rolling mean overlay
    rolling = daily["Energy_kWh"].rolling(30).mean()
    ax.plot(daily.index, rolling, color=PALETTE[1], lw=2, label="30-day rolling mean")
    ax.set_title("Daily Energy Consumption (kWh)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return _save(fig, "daily_trend")


# ── 2. Hourly average profile ────────────────────────────────────────────────

def plot_hourly_profile(hourly: pd.DataFrame) -> plt.Figure:
    hourly_avg = hourly.groupby(hourly.index.hour)["Global_active_power"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(hourly_avg.index, hourly_avg.values, color=PALETTE[2], edgecolor="white")
    ax.set_title("Average Power Consumption by Hour of Day (kW)", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Global Active Power (kW)")
    ax.set_xticks(range(24))
    # Annotate peak
    peak_hour = hourly_avg.idxmax()
    ax.bar(peak_hour, hourly_avg[peak_hour], color=PALETTE[3], label=f"Peak: {peak_hour}:00")
    ax.legend()
    return _save(fig, "hourly_profile")


# ── 3. Weekday vs Weekend ────────────────────────────────────────────────────

def plot_weekday_weekend(hourly: pd.DataFrame) -> plt.Figure:
    df = hourly.copy()
    df["hour"] = df.index.hour
    df["is_weekend"] = (df.index.weekday >= 5).astype(int)

    wd = df[df["is_weekend"] == 0].groupby("hour")["Global_active_power"].mean()
    we = df[df["is_weekend"] == 1].groupby("hour")["Global_active_power"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wd.index, wd.values, marker="o", label="Weekday", color=PALETTE[0])
    ax.plot(we.index, we.values, marker="s", label="Weekend", color=PALETTE[1])
    ax.set_title("Avg Hourly Power – Weekday vs Weekend (kW)", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Global Active Power (kW)")
    ax.set_xticks(range(24))
    ax.legend()
    return _save(fig, "weekday_weekend")


# ── 4. Monthly seasonality ────────────────────────────────────────────────────

def plot_monthly_seasonality(daily: pd.DataFrame) -> plt.Figure:
    df = daily.copy()
    df["month"] = df.index.month
    monthly = df.groupby("month")["Energy_kWh"].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.index, monthly.values, color=PALETTE[4], edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title("Average Daily Energy by Month (kWh)", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Energy (kWh)")
    return _save(fig, "monthly_seasonality")


# ── 5. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame, title: str = "Feature Correlation") -> plt.Figure:
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, vmin=-1, vmax=1
    )
    ax.set_title(title, fontsize=14)
    return _save(fig, "correlation_heatmap")


# ── 6. Sub-metering breakdown ─────────────────────────────────────────────────

def plot_submetering(daily: pd.DataFrame) -> plt.Figure:
    cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    monthly = daily.copy()
    monthly["month"] = monthly.index.month
    agg = monthly.groupby("month")[cols].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.25
    x = np.arange(1, 13)
    ax.bar(x - width, agg["Sub_metering_1"], width, label="Kitchen (SM1)", color=PALETTE[0])
    ax.bar(x,          agg["Sub_metering_2"], width, label="Laundry (SM2)", color=PALETTE[1])
    ax.bar(x + width,  agg["Sub_metering_3"], width, label="HVAC (SM3)",    color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_title("Monthly Average Sub-Metering (Wh)", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Wh")
    ax.legend()
    return _save(fig, "submetering")


# ── 7. Distribution of power consumption ─────────────────────────────────────

def plot_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = df["Global_active_power"].dropna()
    axes[0].hist(data, bins=60, color=PALETTE[5], edgecolor="white")
    axes[0].set_title("Distribution of Global Active Power (kW)")
    axes[0].set_xlabel("kW")
    axes[0].set_ylabel("Frequency")

    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[6], color="black"))
    axes[1].set_title("Box Plot – Global Active Power (kW)")
    axes[1].set_ylabel("kW")
    return _save(fig, "distribution")


# ── 8. Year-over-year comparison ──────────────────────────────────────────────

def plot_yoy_comparison(daily: pd.DataFrame) -> plt.Figure:
    df = daily.copy()
    df["year"] = df.index.year
    df["day_of_year"] = df.index.day_of_year
    years = sorted(df["year"].unique())

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, yr in enumerate(years):
        subset = df[df["year"] == yr]
        ax.plot(subset["day_of_year"], subset["Energy_kWh"],
                lw=1, alpha=0.75, label=str(yr), color=PALETTE[i % len(PALETTE)])
    ax.set_title("Year-over-Year Daily Energy Comparison (kWh)", fontsize=14)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Energy (kWh)")
    ax.legend(title="Year", loc="upper left", ncol=2)
    return _save(fig, "yoy_comparison")
