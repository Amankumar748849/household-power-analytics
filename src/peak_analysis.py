"""
peak_analysis.py
----------------
Step 9 - Use forecasts to identify expected peak periods and support
         energy-management decisions.

Flow
----
  Historical data  ->  top-5 peak hours (by mean)
  ML predictions   ->  90th-pct threshold  ->  peak flags
  Union of both    ->  per-hour demand table
                   ->  structured recommendations (load-shift / HVAC / battery / voltage)
                   ->  static plots

Public API
----------
  build_hourly_demand_table   - full hour-by-hour summary (historical + forecast)
  detect_historical_peaks     - top-N rows from actuals
  forecast_peaks              - tag predictions above the 90th-pct threshold
  energy_management_report    - structured recommendation dict
  plot_peak_heatmap           - hour x weekday heatmap
  plot_forecast_with_peaks    - prediction time-series with peak markers
  plot_voltage_band           - voltage time-series with 220-240 V safety band
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

VOLTAGE_LOW  = 220.0
VOLTAGE_HIGH = 240.0

# ---------------------------------------------------------------------------
# 1. Historical peak detection
# ---------------------------------------------------------------------------

def detect_historical_peaks(
    hourly: pd.DataFrame,
    top_n: int = 10,
    column: str = "Global_active_power",
) -> pd.DataFrame:
    """Return the top-N individual hourly records sorted by consumption."""
    peaks = (
        hourly[[column]]
        .copy()
        .assign(
            hour    = hourly.index.hour,
            weekday = hourly.index.day_name(),
            date    = hourly.index.date,
        )
        .sort_values(column, ascending=False)
        .head(top_n)
    )
    return peaks


def build_hourly_demand_table(
    hourly: pd.DataFrame,
    column: str = "Global_active_power",
) -> pd.DataFrame:
    """
    Build a 24-row table (one per hour) with:
      hour, avg_kw, max_kw, std_kw,
      pct_of_day_total,
      is_historical_peak (top-5),
      demand_label (Low / Medium / High / Peak)
    """
    df = hourly.copy()
    df["hour"] = df.index.hour

    agg = df.groupby("hour")[column].agg(
        avg_kw="mean", max_kw="max", std_kw="std"
    ).reset_index()

    total = agg["avg_kw"].sum()
    agg["pct_of_day"] = (agg["avg_kw"] / total * 100).round(1)

    # Mark historical top-5 hours
    top5 = agg.nlargest(5, "avg_kw")["hour"].tolist()
    agg["is_historical_peak"] = agg["hour"].isin(top5)

    # Demand label via quartile buckets
    q25 = agg["avg_kw"].quantile(0.25)
    q50 = agg["avg_kw"].quantile(0.50)
    q75 = agg["avg_kw"].quantile(0.75)

    def _label(v):
        if v >= q75:
            return "Peak"
        if v >= q50:
            return "High"
        if v >= q25:
            return "Medium"
        return "Low"

    agg["demand_label"] = agg["avg_kw"].apply(_label)
    agg = agg.sort_values("hour").reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# 2. Forecast peak tagging
# ---------------------------------------------------------------------------

def forecast_peaks(
    test_index: pd.DatetimeIndex,
    y_pred: np.ndarray,
    threshold_quantile: float = 0.90,
) -> tuple[pd.DataFrame, float]:
    """
    Tag predicted values at or above the given quantile as peak.

    Returns
    -------
    df        : DataFrame with columns [predicted, is_peak]
    threshold : float  - the 90th-pct value used
    """
    threshold = float(np.quantile(y_pred, threshold_quantile))
    df = pd.DataFrame(
        {"predicted": y_pred},
        index=test_index[: len(y_pred)],
    )
    df.index.name = "datetime"
    df["is_peak"] = df["predicted"] >= threshold
    return df, threshold


def build_forecast_hour_summary(
    forecast_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Summarise the forecast by hour:
      hour, avg_predicted, peak_count, peak_pct, is_forecast_peak (top-5 by freq)
    """
    df = forecast_df.copy()
    df["hour"] = df.index.hour

    agg = df.groupby("hour").agg(
        avg_predicted=("predicted", "mean"),
        peak_count=("is_peak", "sum"),
        total_count=("is_peak", "count"),
    ).reset_index()
    agg["peak_pct"] = (agg["peak_count"] / agg["total_count"] * 100).round(1)

    top5_fc = agg.nlargest(5, "peak_count")["hour"].tolist()
    agg["is_forecast_peak"] = agg["hour"].isin(top5_fc)
    return agg.sort_values("hour").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Energy-management recommendation engine
# ---------------------------------------------------------------------------

# Which hours are considered "night off-peak" for battery pre-charge
_OFF_PEAK_HOURS = list(range(1, 7))      # 01:00-06:00

# HVAC is sub-metering 3; Kitchen SM1; Laundry SM2
_LOAD_LABELS = {
    "Sub_metering_1": "Kitchen appliances",
    "Sub_metering_2": "Laundry / washing machine",
    "Sub_metering_3": "HVAC (heating / cooling)",
}


def energy_management_report(
    hourly: pd.DataFrame,
    forecast_df: pd.DataFrame,
    threshold: float,
    demand_table: pd.DataFrame = None,
    fc_hour_summary: pd.DataFrame = None,
) -> dict:
    """
    Produce a fully structured energy-management dict.

    Keys
    ----
    peak_hours_historical  : list[int]   top-5 historical hours
    peak_hours_forecast    : list[int]   top-5 forecasted peak hours
    union_peak_hours       : list[int]   sorted union of both sets
    threshold              : float
    stats                  : dict        mean/max/threshold/peak_count
    recommendations        : list[dict]  typed recommendation cards
      each card: {category, icon, title, detail, hours}
    voltage_alerts         : dict        pct of readings outside 220-240 V
    """

    # --- historical top-5 peak hours ------------------------------------
    h_avg = hourly.groupby(hourly.index.hour)["Global_active_power"].mean()
    top_hist = h_avg.nlargest(5).index.tolist()

    # --- forecasted top-5 peak hours ------------------------------------
    df_fc = forecast_df.copy()
    df_fc["hour"] = df_fc.index.hour
    top_fc = (
        df_fc[df_fc["is_peak"]]
        ["hour"].value_counts().head(5).index.tolist()
    )

    union_hours = sorted(set(top_hist) | set(top_fc))

    # --- voltage analysis -----------------------------------------------
    if "Voltage" in hourly.columns:
        v = hourly["Voltage"].dropna()
        n_total = len(v)
        n_low   = int((v < VOLTAGE_LOW).sum())
        n_high  = int((v > VOLTAGE_HIGH).sum())
        pct_low  = round(n_low  / n_total * 100, 2) if n_total else 0
        pct_high = round(n_high / n_total * 100, 2) if n_total else 0
    else:
        pct_low = pct_high = 0

    # --- build recommendation cards ------------------------------------
    recs = []

    # Card 1 – Load Shifting
    hour_strs = ", ".join(f"{h:02d}:00" for h in sorted(union_hours))
    recs.append({
        "category": "load_shift",
        "icon": "SHIFT",
        "title": "Shift Flexible Loads to Off-Peak",
        "detail": (
            f"High demand is expected around {hour_strs}. "
            "Move washing-machine cycles, dishwasher runs, and EV charging "
            "to off-peak windows (01:00-06:00) to reduce peak grid stress."
        ),
        "hours": sorted(union_hours),
    })

    # Card 2 – HVAC scheduling
    hvac_hours = [h for h in union_hours if h in range(6, 23)]
    recs.append({
        "category": "hvac",
        "icon": "HVAC",
        "title": "Pre-condition HVAC Before Peak Windows",
        "detail": (
            "Sub-metering 3 (HVAC) is the largest flexible load. "
            f"Start pre-cooling / pre-heating 30-60 minutes before {hour_strs} "
            "so the system can coast during the peak without full power draw."
        ),
        "hours": hvac_hours,
    })

    # Card 3 – Battery / storage
    recs.append({
        "category": "battery",
        "icon": "BATT",
        "title": "Pre-charge Storage During Off-Peak (01:00-06:00)",
        "detail": (
            "If a home battery or EV with V2H capability is available, "
            "schedule charging between 01:00 and 06:00 when average demand is lowest. "
            f"Discharge during the predicted peak window ({hour_strs}) "
            "to reduce grid import and lower time-of-use tariff costs."
        ),
        "hours": _OFF_PEAK_HOURS,
    })

    # Card 4 – Voltage monitoring
    volt_detail = (
        f"Target: {VOLTAGE_LOW}-{VOLTAGE_HIGH} V. "
        f"Historical out-of-band readings: {pct_low}% below {VOLTAGE_LOW} V, "
        f"{pct_high}% above {VOLTAGE_HIGH} V. "
        "Voltage sags often coincide with peak-demand periods - "
        "enable alerts on your smart meter or energy monitor."
    )
    recs.append({
        "category": "voltage",
        "icon": "VOLT",
        "title": f"Monitor Voltage During Peak Hours ({hour_strs})",
        "detail": volt_detail,
        "hours": sorted(union_hours),
    })

    # Card 5 – Tariff / demand response
    recs.append({
        "category": "tariff",
        "icon": "COST",
        "title": "Review Time-of-Use Tariff Alignment",
        "detail": (
            f"Peak demand clusters at {hour_strs}. "
            "Check whether your energy supplier's peak tariff windows overlap "
            "with these hours. Even a 30-minute shift in large appliance usage "
            "can meaningfully reduce your monthly electricity bill."
        ),
        "hours": sorted(union_hours),
    })

    return {
        "peak_hours_historical": top_hist,
        "peak_hours_forecast":   top_fc,
        "union_peak_hours":      union_hours,
        "threshold":             round(threshold, 4),
        "stats": {
            "mean_predicted": round(float(forecast_df["predicted"].mean()), 4),
            "max_predicted":  round(float(forecast_df["predicted"].max()),  4),
            "threshold":      round(float(threshold), 4),
            "peak_count":     int(forecast_df["is_peak"].sum()),
            "pct_peaks":      round(forecast_df["is_peak"].mean() * 100, 1),
        },
        "voltage_alerts": {
            "pct_below_220": pct_low,
            "pct_above_240": pct_high,
        },
        "recommendations": recs,
    }


# ---------------------------------------------------------------------------
# 4. Static plots (for main.py / offline use)
# ---------------------------------------------------------------------------

def plot_peak_heatmap(hourly: pd.DataFrame) -> plt.Figure:
    """Hour x Weekday heatmap of average Global Active Power."""
    df = hourly.copy()
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.weekday

    pivot = df.pivot_table(
        values="Global_active_power",
        index="weekday",
        columns="hour",
        aggfunc="mean",
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot, cmap="YlOrRd", linewidths=0.3, ax=ax,
        cbar_kws={"label": "Avg kW"},
    )
    ax.set_title("Average Power Consumption - Hour x Weekday Heatmap", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "peak_heatmap.png", dpi=150)
    return fig


def plot_forecast_with_peaks(
    forecast_df: pd.DataFrame,
    threshold: float,
    model_name: str = "Best Model",
    max_points: int = 500,
) -> plt.Figure:
    """Time-series of predictions with peak periods highlighted."""
    df = forecast_df.iloc[:max_points].copy()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["predicted"], color="#1f77b4", lw=1, label="Predicted")
    peaks = df[df["is_peak"]]
    ax.scatter(peaks.index, peaks["predicted"],
               color="red", s=20, zorder=5,
               label=f"Peak (>= {threshold:.2f} kW  |  90th pct)")
    ax.axhline(threshold, color="orange", ls="--", lw=1.5, label="Peak threshold")
    ax.fill_between(df.index, threshold, df["predicted"],
                    where=df["predicted"] >= threshold,
                    alpha=0.15, color="red", label="Peak zone")
    ax.set_title(f"{model_name} - Forecasted Consumption with Peak Markers", fontsize=14)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend()
    fig.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    fig.savefig(PLOT_DIR / f"forecast_peaks_{safe}.png", dpi=150)
    return fig


def plot_voltage_band(hourly: pd.DataFrame, max_points: int = 2000) -> plt.Figure:
    """
    Voltage time-series with a 220-240 V safety band highlighted.
    Out-of-band readings are marked in red.
    """
    if "Voltage" not in hourly.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Voltage column", ha="center", va="center")
        return fig

    sample = hourly["Voltage"].dropna().iloc[:max_points]
    in_band  = sample.where((sample >= VOLTAGE_LOW) & (sample <= VOLTAGE_HIGH))
    out_band = sample.where((sample < VOLTAGE_LOW) | (sample > VOLTAGE_HIGH))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sample.index, in_band,  color="#2196F3", lw=0.8, label="Normal (220-240 V)")
    ax.scatter(out_band.dropna().index, out_band.dropna(),
               color="red", s=8, zorder=5, label="Out of band")
    ax.axhspan(VOLTAGE_LOW, VOLTAGE_HIGH, alpha=0.08, color="green", label="Safe band")
    ax.axhline(VOLTAGE_LOW,  color="green", ls="--", lw=1)
    ax.axhline(VOLTAGE_HIGH, color="green", ls="--", lw=1)
    ax.set_title("Voltage Time-Series with 220-240 V Safety Band", fontsize=14)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "voltage_band.png", dpi=150)
    return fig


def plot_hourly_demand_bar(demand_table: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of avg kW per hour, coloured by demand_label.
    """
    colour_map = {"Low": "#4caf50", "Medium": "#ff9800",
                  "High": "#f44336", "Peak": "#b71c1c"}
    colours = demand_table["demand_label"].map(colour_map).fillna("#9e9e9e")

    fig, ax = plt.subplots(figsize=(13, 4))
    bars = ax.bar(demand_table["hour"], demand_table["avg_kw"],
                  color=colours, edgecolor="white", width=0.8)
    ax.set_xticks(demand_table["hour"])
    ax.set_xticklabels([f"{h:02d}:00" for h in demand_table["hour"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title("Average Power by Hour - Demand Level", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg kW")

    patches = [mpatches.Patch(color=v, label=k) for k, v in colour_map.items()]
    ax.legend(handles=patches, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hourly_demand_bar.png", dpi=150)
    return fig
