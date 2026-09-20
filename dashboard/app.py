"""
dashboard/app.py
────────────────
Streamlit interactive dashboard for Household Power Consumption Analysis.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from src.data_processing import run_pipeline
from src.features import build_hourly_features, build_daily_features
from src.peak_analysis import (
    detect_historical_peaks,
    build_hourly_demand_table,
    forecast_peaks,
    build_forecast_hour_summary,
    energy_management_report,
    VOLTAGE_LOW,
    VOLTAGE_HIGH,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⚡ Household Power Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Main background */
.main { background-color: #0f1117; }
/* KPI card */
.kpi-box {
    background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
    border: 1px solid #2e3459;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4facfe;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.8rem;
    color: #9aa0b4;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kpi-delta-good { color: #00d97e; font-size: 0.85rem; }
.kpi-delta-bad  { color: #ff6b6b; font-size: 0.85rem; }
/* Section header */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #c9d1e8;
    border-left: 4px solid #4facfe;
    padding-left: 12px;
    margin: 1.2rem 0 0.8rem;
}
/* Recommendation card */
.rec-card {
    background: #1a1d2e;
    border: 1px solid #2e3459;
    border-left: 4px solid #f7b731;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: #d0d4e8;
    font-size: 0.9rem;
}
/* Metric comparison table */
.metrics-table th { color: #4facfe !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#4facfe"
ORANGE = "#ff7f0e"
GREEN  = "#00d97e"
RED    = "#ff6b6b"

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & processing dataset …")
def load_data():
    return run_pipeline()


@st.cache_data(show_spinner="Engineering features …")
def get_hourly_features(_hourly):
    return build_hourly_features(_hourly)


@st.cache_data(show_spinner="Engineering daily features …")
def get_daily_features(_daily):
    return build_daily_features(_daily)


# ── Load models (if available) ────────────────────────────────────────────────

def try_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/lightning-bolt.png", width=60)
    st.title("⚡ Power Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📈 Trends & EDA",
            "🔍 Deep Dive",
            "🤖 Model Results",
            "📍 Peak Analysis",
            "📋 Data Quality",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")
    data = load_data()
    minute_df = data["minute"]
    hourly_df  = data["hourly"]
    daily_df   = data["daily"]
    qa_report  = data["report"]

    min_date = daily_df.index.min().date()
    max_date = daily_df.index.max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(date_range) == 2:
        d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        d_start, d_end = pd.Timestamp(min_date), pd.Timestamp(max_date)

    hourly_f = hourly_df.loc[d_start:d_end]
    daily_f  = daily_df.loc[d_start:d_end]

    st.markdown("---")
    st.caption("Dataset: UCI Household Power Consumption")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: KPI card
# ─────────────────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, delta: str = "", good: bool = True):
    delta_class = "kpi-delta-good" if good else "kpi-delta-bad"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("⚡ Household Power Consumption Analytics")
    st.markdown(
        "Real-time analytics dashboard for a multi-year household electricity dataset. "
        "Use the sidebar to filter by date range and navigate sections."
    )

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    total_kwh  = daily_f["Energy_kWh"].sum()
    avg_daily  = daily_f["Energy_kWh"].mean()
    peak_power = hourly_f["Global_active_power"].max()
    avg_volt   = hourly_f["Voltage"].mean()
    n_days     = len(daily_f)

    with col1: kpi("Total Energy", f"{total_kwh:,.0f} kWh")
    with col2: kpi("Avg Daily Usage", f"{avg_daily:.1f} kWh")
    with col3: kpi("Peak Power", f"{peak_power:.2f} kW")
    with col4: kpi("Avg Voltage", f"{avg_volt:.1f} V")
    with col5: kpi("Days in Range", f"{n_days:,}")

    st.markdown("---")

    # Daily energy trend (Plotly)
    section("Daily Energy Consumption (kWh)")
    roll30 = daily_f["Energy_kWh"].rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_f.index, y=daily_f["Energy_kWh"],
        mode="lines", name="Daily Energy",
        line=dict(color=ACCENT, width=1), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=daily_f.index, y=roll30,
        mode="lines", name="30-day MA",
        line=dict(color=ORANGE, width=2.5)
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=300,
        xaxis_title="Date", yaxis_title="Energy (kWh)",
        margin=dict(l=40, r=20, t=20, b=40), legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sub-metering donut
    col_a, col_b = st.columns(2)
    with col_a:
        section("Sub-Metering Share (Selected Period)")
        sm_totals = {
            "Kitchen (SM1)": daily_f["Sub_metering_1"].sum(),
            "Laundry (SM2)": daily_f["Sub_metering_2"].sum(),
            "HVAC (SM3)":    daily_f["Sub_metering_3"].sum(),
        }
        fig2 = go.Figure(go.Pie(
            labels=list(sm_totals.keys()),
            values=list(sm_totals.values()),
            hole=0.5,
            marker_colors=["#4facfe", "#f7b731", "#ff6b6b"],
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        section("Monthly Average Power (kW)")
        monthly = hourly_f.copy()
        monthly["month"] = monthly.index.month
        m_avg = monthly.groupby("month")["Global_active_power"].mean()
        m_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig3 = go.Figure(go.Bar(
            x=[m_labels[i-1] for i in m_avg.index],
            y=m_avg.values,
            marker_color=ACCENT,
        ))
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            xaxis_title="Month", yaxis_title="Avg Power (kW)",
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRENDS & EDA
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Trends & EDA":
    st.title("📈 Exploratory Data Analysis")

    # Hourly profile
    section("Average Consumption by Hour of Day")
    h_avg = hourly_f.groupby(hourly_f.index.hour)["Global_active_power"].mean()
    fig = go.Figure(go.Bar(
        x=h_avg.index, y=h_avg.values,
        marker_color=[RED if v == h_avg.max() else ACCENT for v in h_avg.values],
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        xaxis=dict(title="Hour", tickmode="linear"),
        yaxis_title="Avg kW",
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weekday vs Weekend
    section("Weekday vs Weekend Hourly Profile")
    df_h = hourly_f.copy()
    df_h["hour"] = df_h.index.hour
    df_h["is_weekend"] = (df_h.index.weekday >= 5)
    wd = df_h[~df_h["is_weekend"]].groupby("hour")["Global_active_power"].mean()
    we = df_h[df_h["is_weekend"]].groupby("hour")["Global_active_power"].mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=wd.index, y=wd.values, mode="lines+markers",
                              name="Weekday", line=dict(color=ACCENT)))
    fig2.add_trace(go.Scatter(x=we.index, y=we.values, mode="lines+markers",
                              name="Weekend", line=dict(color=ORANGE)))
    fig2.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        xaxis=dict(title="Hour", tickmode="linear"),
        yaxis_title="Avg kW",
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Power Distribution")
        data_dist = hourly_f["Global_active_power"].dropna()
        fig3 = px.histogram(data_dist, nbins=80, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=[ACCENT])
        fig3.update_layout(
            height=300, xaxis_title="Global Active Power (kW)",
            yaxis_title="Count", margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        section("Seasonal Box Plot (by Month)")
        df_box = hourly_f.copy()
        df_box["month"] = df_box.index.month
        m_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        df_box["Month"] = df_box["month"].map(m_labels)
        fig4 = px.box(df_box, x="Month", y="Global_active_power",
                      category_orders={"Month": list(m_labels.values())},
                      template=PLOTLY_TEMPLATE,
                      color_discrete_sequence=[ACCENT])
        fig4.update_layout(
            height=300, yaxis_title="kW",
            margin=dict(l=40, r=20, t=20, b=40), showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation matrix
    section("Feature Correlation Matrix")
    corr_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
                 "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    corr_cols = [c for c in corr_cols if c in hourly_f.columns]
    corr = hourly_f[corr_cols].corr().round(2)
    fig5 = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        template=PLOTLY_TEMPLATE, zmin=-1, zmax=1,
        aspect="auto"
    )
    fig5.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Deep Dive":
    st.title("🔍 Deep Dive Analysis")

    # Year-over-year
    section("Year-over-Year Daily Energy (kWh)")
    df_yoy = daily_f.copy()
    df_yoy["year"] = df_yoy.index.year
    df_yoy["doy"]  = df_yoy.index.day_of_year
    fig = px.line(df_yoy, x="doy", y="Energy_kWh", color="year",
                  template=PLOTLY_TEMPLATE,
                  labels={"doy": "Day of Year", "Energy_kWh": "Energy (kWh)", "year": "Year"})
    fig.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap hour × weekday
    section("Consumption Heatmap – Hour × Day of Week")
    df_heat = hourly_f.copy()
    df_heat["hour"]    = df_heat.index.hour
    df_heat["weekday"] = df_heat.index.weekday
    pivot = df_heat.pivot_table(
        values="Global_active_power", index="weekday",
        columns="hour", aggfunc="mean"
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig2 = px.imshow(
        pivot, color_continuous_scale="YlOrRd",
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg kW"),
        template=PLOTLY_TEMPLATE, aspect="auto"
    )
    fig2.update_layout(height=300, margin=dict(l=60, r=20, t=20, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Sub-Metering Monthly Trend")
        sm_daily = daily_f.copy()
        sm_daily["month"]    = sm_daily.index.month
        sm_daily["year"]     = sm_daily.index.year
        sm_monthly = sm_daily.groupby(["year","month"])[
            ["Sub_metering_1","Sub_metering_2","Sub_metering_3"]
        ].mean().reset_index()
        sm_monthly["period"] = sm_monthly["year"].astype(str) + "-" + sm_monthly["month"].astype(str).str.zfill(2)
        fig3 = go.Figure()
        for col_sm, name, color in [
            ("Sub_metering_1", "Kitchen (SM1)", ACCENT),
            ("Sub_metering_2", "Laundry (SM2)", ORANGE),
            ("Sub_metering_3", "HVAC (SM3)",    RED),
        ]:
            fig3.add_trace(go.Scatter(
                x=sm_monthly["period"], y=sm_monthly[col_sm],
                mode="lines", name=name, line=dict(color=color)
            ))
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            xaxis_title="Period", yaxis_title="Avg Wh",
            margin=dict(l=40, r=20, t=20, b=60),
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        section("Voltage vs Power Scatter")
        sample = hourly_f.sample(min(2000, len(hourly_f)), random_state=42)
        fig4 = px.scatter(
            sample, x="Voltage", y="Global_active_power",
            color="Global_intensity", color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE, opacity=0.6,
            labels={"Global_active_power": "Active Power (kW)", "Global_intensity": "Intensity (A)"},
        )
        fig4.update_layout(
            height=300, margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Rolling stats
    section("7-Day Rolling Mean vs Actual (Daily Energy)")
    roll7  = daily_f["Energy_kWh"].rolling(7).mean()
    roll30 = daily_f["Energy_kWh"].rolling(30).mean()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=daily_f.index, y=daily_f["Energy_kWh"],
                              mode="lines", name="Daily", opacity=0.4,
                              line=dict(color=ACCENT, width=1)))
    fig5.add_trace(go.Scatter(x=daily_f.index, y=roll7,
                              mode="lines", name="7-day MA",
                              line=dict(color=ORANGE, width=2)))
    fig5.add_trace(go.Scatter(x=daily_f.index, y=roll30,
                              mode="lines", name="30-day MA",
                              line=dict(color=GREEN, width=2)))
    fig5.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        xaxis_title="Date", yaxis_title="Energy (kWh)",
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 Model Results":
    st.title("🤖 Forecasting Model Results")

    MODEL_DIR  = Path("outputs/models")
    REPORT_DIR = Path("outputs/reports")

    # Load metrics CSV if available
    metrics_path = REPORT_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, index_col=0)

        section("Model Comparison – MAE / RMSE / R²")
        col1, col2, col3 = st.columns(3)
        for ax_col, metric, lower_better, color in [
            (col1, "MAE",  True,  ACCENT),
            (col2, "RMSE", True,  ORANGE),
            (col3, "R2",   False, GREEN),
        ]:
            if metric in metrics_df.columns:
                with ax_col:
                    fig = go.Figure(go.Bar(
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        marker_color=color,
                        text=metrics_df[metric].round(4),
                        textposition="outside"
                    ))
                    fig.update_layout(
                        template=PLOTLY_TEMPLATE, height=280, title=metric,
                        margin=dict(l=20, r=20, t=40, b=60),
                        xaxis=dict(tickangle=15),
                        yaxis_title=metric
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        section("Detailed Metrics Table")
        styled = metrics_df.style\
            .background_gradient(subset=["MAE","RMSE"], cmap="RdYlGn_r")\
            .background_gradient(subset=["R2"], cmap="RdYlGn")\
            .format(precision=4)
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("ℹ️ Run `python main.py --no-lstm` first to train models and generate metrics.")

    # Prediction plots
    plot_files = sorted(Path("outputs/plots").glob("pred_*.png"))
    if plot_files:
        section("Actual vs Predicted Plots")
        for pf in plot_files:
            st.image(str(pf), caption=pf.stem.replace("pred_", "").replace("_", " ").title(),
                     use_container_width=True)
    else:
        st.info("ℹ️ Prediction plots will appear here after training.")

    # Feature importance plots
    imp_files = sorted(Path("outputs/plots").glob("importance_*.png"))
    if imp_files:
        section("Feature Importance")
        cols = st.columns(len(imp_files))
        for col_fi, pf in zip(cols, imp_files):
            with col_fi:
                st.image(str(pf), caption=pf.stem.replace("importance_", "").replace("_", " ").title(),
                         use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PEAK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📍 Peak Analysis":
    import json as _json

    st.title("📍 Peak Detection & Energy Management")
    st.markdown(
        "This page traces the full energy-management pipeline — "
        "from raw historical consumption through ML forecasting to actionable recommendations."
    )

    # ── FLOW DIAGRAM ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1a1d2e;border:1px solid #2e3459;border-radius:12px;
                padding:18px 24px;margin-bottom:20px;font-size:0.88rem;color:#c9d1e8;
                line-height:2;">
      <b style="color:#4facfe;">Energy Management Flow</b><br>
      <span style="color:#9aa0b4;">Historical Data</span>
      &nbsp;&#8594;&nbsp; <span style="color:#f7b731;">Find High-Consumption Hours</span>
      &nbsp;&#8594;&nbsp; <span style="color:#4facfe;">ML Model Predicts Future Consumption</span>
      &nbsp;&#8594;&nbsp; <span style="color:#f7b731;">Calculate 90th-Pct Peak Threshold</span>
      &nbsp;&#8594;&nbsp; <span style="color:#ff6b6b;">Identify Predicted Peak Hours</span>
      &nbsp;&#8594;&nbsp; <span style="color:#00d97e;">Generate Recommendations</span>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 1: HOURLY DEMAND TABLE ───────────────────────────────────────────
    section("Step 1 — Historical Hourly Demand Profile")
    demand_table = build_hourly_demand_table(hourly_f)

    # Colour-coded bar chart
    DEMAND_COLORS = {"Low": "#4caf50", "Medium": "#ff9800",
                     "High": "#f44336", "Peak": "#8b0000"}
    bar_colors = demand_table["demand_label"].map(DEMAND_COLORS).tolist()

    fig_dem = go.Figure()
    fig_dem.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in demand_table["hour"]],
        y=demand_table["avg_kw"],
        marker_color=bar_colors,
        text=demand_table["demand_label"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.3f} kW<br>Max: %{customdata[0]:.3f} kW<br>Share: %{customdata[1]:.1f}%<extra></extra>",
        customdata=demand_table[["max_kw", "pct_of_day"]].values,
    ))
    # Mark historical top-5 with vertical annotation
    top5_hours = demand_table[demand_table["is_historical_peak"]]["hour"].tolist()
    for h in top5_hours:
        fig_dem.add_vline(x=f"{h:02d}:00", line_dash="dot",
                          line_color="#f7b731", line_width=1.5)
    threshold_90_hist = float(np.quantile(hourly_f["Global_active_power"].dropna(), 0.90))
    fig_dem.add_hline(y=threshold_90_hist, line_dash="dash", line_color="#ff6b6b",
                      annotation_text=f"90th pct = {threshold_90_hist:.2f} kW",
                      annotation_position="top right")
    fig_dem.update_layout(
        template=PLOTLY_TEMPLATE, height=360,
        xaxis_title="Hour of Day", yaxis_title="Avg kW",
        margin=dict(l=40, r=20, t=20, b=60),
        showlegend=False,
    )
    st.plotly_chart(fig_dem, use_container_width=True)

    # Expandable table
    with st.expander("View full 24-hour demand table"):
        display_cols = ["hour", "avg_kw", "max_kw", "std_kw", "pct_of_day", "demand_label", "is_historical_peak"]
        st.dataframe(
            demand_table[display_cols].style
            .background_gradient(subset=["avg_kw", "pct_of_day"], cmap="YlOrRd")
            .applymap(lambda v: "color:#f7b731;font-weight:700" if v is True else "",
                      subset=["is_historical_peak"])
            .format({"avg_kw": "{:.3f}", "max_kw": "{:.3f}",
                     "std_kw": "{:.3f}", "pct_of_day": "{:.1f}%"}),
            use_container_width=True,
        )

    # ── STEP 2: HEATMAP ───────────────────────────────────────────────────────
    section("Step 2 — Hour x Weekday Heatmap (Historical Actuals)")
    df_heat = hourly_f.copy()
    df_heat["hour"]    = df_heat.index.hour
    df_heat["weekday"] = df_heat.index.weekday
    pivot = df_heat.pivot_table(
        values="Global_active_power", index="weekday",
        columns="hour", aggfunc="mean"
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_hm = px.imshow(
        pivot, color_continuous_scale="YlOrRd",
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg kW"),
        template=PLOTLY_TEMPLATE, aspect="auto",
        zmin=0,
    )
    fig_hm.update_layout(height=300, margin=dict(l=60, r=20, t=20, b=40))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── STEP 3: FORECAST PEAKS ────────────────────────────────────────────────
    section("Step 3 — 90th-Percentile Peak Threshold & Forecast Tags")

    REPORT_DIR = Path("outputs/reports")
    mgmt_path  = REPORT_DIR / "energy_management.json"

    if mgmt_path.exists():
        with open(mgmt_path) as _f:
            mgmt = _json.load(_f)

        stats = mgmt["stats"]
        volt  = mgmt.get("voltage_alerts", {})

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi("Mean Predicted", f"{stats['mean_predicted']:.3f} kW")
        with c2: kpi("Max Predicted",  f"{stats['max_predicted']:.3f} kW")
        with c3: kpi("Peak Threshold (90th pct)", f"{stats['threshold']:.3f} kW")
        with c4: kpi("Peak Count", f"{stats['peak_count']:,}",
                     delta=f"{stats['pct_peaks']:.1f}% of test set",
                     good=(stats['pct_peaks'] <= 12))
        with c5: kpi("Voltage < 220V", f"{volt.get('pct_below_220', 0):.2f}%",
                     delta="Out-of-band", good=(volt.get('pct_below_220', 0) < 1))

        st.markdown("")

        # Threshold explanation callout
        st.markdown(f"""
        <div style="background:#1e2130;border-left:4px solid #ff6b6b;border-radius:8px;
                    padding:14px 18px;margin-bottom:16px;color:#d0d4e8;font-size:0.9rem;">
          <b style="color:#ff6b6b;">How the threshold works:</b><br>
          The model predicts hourly consumption for the held-out test period.
          The <b>90th percentile</b> of those predictions is calculated:
          <b style="color:#f7b731;">{stats['threshold']:.3f} kW</b>.
          Any hour where predicted consumption exceeds this value is flagged
          as a <b style="color:#ff6b6b;">PEAK</b> period.
          This is adaptive — it adjusts automatically to the model's own output
          distribution rather than using a fixed rule.
        </div>
        """, unsafe_allow_html=True)

        # Forecast plot images
        forecast_peak_imgs = sorted(Path("outputs/plots").glob("forecast_peaks_*.png"))
        if forecast_peak_imgs:
            for fp in forecast_peak_imgs:
                st.image(str(fp),
                         caption=f"Forecast: {fp.stem.replace('forecast_peaks_','').replace('_',' ').title()} — red = peak (>= {stats['threshold']:.3f} kW)",
                         use_container_width=True)

        # ── STEP 4: PREDICTED PEAK HOURS ──────────────────────────────────────
        section("Step 4 — Predicted vs Historical Peak Hours")

        hist_hours = mgmt.get("peak_hours_historical", [])
        fc_hours   = mgmt.get("peak_hours_forecast",   [])
        union_hrs  = mgmt.get("union_peak_hours",       [])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Historical top-5 peak hours (by avg consumption)**")
            h_avg_all = hourly_f.groupby(hourly_f.index.hour)["Global_active_power"].mean()
            fig_hist = go.Figure(go.Bar(
                x=[f"{h:02d}:00" for h in h_avg_all.index],
                y=h_avg_all.values,
                marker_color=[RED if h in hist_hours else ACCENT for h in h_avg_all.index],
                hovertemplate="%{x}: %{y:.3f} kW<extra></extra>",
            ))
            fig_hist.update_layout(
                template=PLOTLY_TEMPLATE, height=280,
                xaxis_title="Hour", yaxis_title="Avg kW",
                margin=dict(l=30, r=10, t=20, b=50),
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            st.markdown("**Forecasted peak-hour frequency (test period)**")
            fc_h_avg = hourly_f.groupby(hourly_f.index.hour)["Global_active_power"].mean()
            fig_fc = go.Figure(go.Bar(
                x=[f"{h:02d}:00" for h in fc_h_avg.index],
                y=fc_h_avg.values,
                marker_color=[RED if h in fc_hours else "#555e7b" for h in fc_h_avg.index],
                hovertemplate="%{x}: %{y:.3f} kW<extra></extra>",
            ))
            for h in union_hrs:
                fig_fc.add_vline(x=f"{h:02d}:00", line_dash="dot",
                                 line_color="#f7b731", line_width=1.5)
            fig_fc.update_layout(
                template=PLOTLY_TEMPLATE, height=280,
                xaxis_title="Hour", yaxis_title="Avg kW",
                margin=dict(l=30, r=10, t=20, b=50),
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        # Union summary tags
        tag_html = " &nbsp; ".join(
            f'<span style="background:#8b0000;color:#fff;border-radius:6px;'
            f'padding:3px 10px;font-size:0.82rem;font-weight:600;">{h:02d}:00</span>'
            for h in union_hrs
        )
        st.markdown(
            f'<p style="color:#9aa0b4;font-size:0.85rem;margin-top:6px;">'
            f'<b style="color:#ff6b6b;">Combined peak window:</b> &nbsp;{tag_html}</p>',
            unsafe_allow_html=True
        )

        # ── STEP 5: VOLTAGE MONITORING ─────────────────────────────────────────
        section("Step 5 — Voltage Monitoring (220-240 V Safety Band)")

        v_data = hourly_f["Voltage"].dropna()
        sample_v = v_data.iloc[:2000]
        in_band  = sample_v.where((sample_v >= VOLTAGE_LOW) & (sample_v <= VOLTAGE_HIGH))
        out_band = sample_v.where((sample_v < VOLTAGE_LOW) | (sample_v > VOLTAGE_HIGH))

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=in_band.dropna().index, y=in_band.dropna(),
            mode="lines", name=f"Normal ({VOLTAGE_LOW}-{VOLTAGE_HIGH} V)",
            line=dict(color="#2196F3", width=0.8),
        ))
        if out_band.dropna().shape[0] > 0:
            fig_v.add_trace(go.Scatter(
                x=out_band.dropna().index, y=out_band.dropna(),
                mode="markers", name="Out of band",
                marker=dict(color=RED, size=4),
            ))
        fig_v.add_hrect(y0=VOLTAGE_LOW, y1=VOLTAGE_HIGH,
                        fillcolor="rgba(0,200,100,0.07)", line_width=0,
                        annotation_text="Safe band", annotation_position="top left")
        fig_v.add_hline(y=VOLTAGE_LOW,  line_dash="dash", line_color=GREEN, line_width=1)
        fig_v.add_hline(y=VOLTAGE_HIGH, line_dash="dash", line_color=GREEN, line_width=1)
        fig_v.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            xaxis_title="Datetime", yaxis_title="Voltage (V)",
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(x=0, y=1),
        )
        st.plotly_chart(fig_v, use_container_width=True)

        # Voltage KPIs
        n_total = len(v_data)
        n_low  = int((v_data < VOLTAGE_LOW).sum())
        n_high = int((v_data > VOLTAGE_HIGH).sum())
        cv1, cv2, cv3, cv4 = st.columns(4)
        with cv1: kpi("Avg Voltage", f"{v_data.mean():.2f} V")
        with cv2: kpi("Min Voltage", f"{v_data.min():.2f} V",
                      delta="Below 220V alert" if v_data.min() < VOLTAGE_LOW else "OK", good=v_data.min() >= VOLTAGE_LOW)
        with cv3: kpi("Readings < 220V", f"{n_low:,}",
                      delta=f"{n_low/n_total*100:.2f}% of total", good=(n_low == 0))
        with cv4: kpi("Readings > 240V", f"{n_high:,}",
                      delta=f"{n_high/n_total*100:.2f}% of total", good=(n_high == 0))

        # ── STEP 6: RECOMMENDATION CARDS ──────────────────────────────────────
        section("Step 6 — Energy Management Recommendations")

        CARD_META = {
            "load_shift": {"color": "#f7b731", "emoji": "SHIFT",
                           "bg": "rgba(247,183,49,0.08)",  "border": "#f7b731"},
            "hvac":       {"color": "#4facfe", "emoji": "HVAC",
                           "bg": "rgba(79,172,254,0.08)",  "border": "#4facfe"},
            "battery":    {"color": "#00d97e", "emoji": "BATT",
                           "bg": "rgba(0,217,126,0.08)",   "border": "#00d97e"},
            "voltage":    {"color": "#ff6b6b", "emoji": "VOLT",
                           "bg": "rgba(255,107,107,0.08)", "border": "#ff6b6b"},
            "tariff":     {"color": "#a78bfa", "emoji": "COST",
                           "bg": "rgba(167,139,250,0.08)", "border": "#a78bfa"},
        }

        for rec in mgmt["recommendations"]:
            meta  = CARD_META.get(rec["category"], CARD_META["load_shift"])
            hours_html = " ".join(
                f'<span style="background:{meta["color"]}22;color:{meta["color"]};'
                f'border-radius:4px;padding:1px 7px;font-size:0.78rem;">{h:02d}:00</span>'
                for h in rec.get("hours", [])
            )
            st.markdown(f"""
            <div style="background:{meta['bg']};border:1px solid {meta['border']};
                        border-left:4px solid {meta['border']};border-radius:10px;
                        padding:14px 18px;margin-bottom:12px;">
              <div style="font-size:0.95rem;font-weight:700;color:{meta['color']};margin-bottom:6px;">
                [{rec['icon']}] &nbsp; {rec['title']}
              </div>
              <div style="color:#c9d1e8;font-size:0.875rem;line-height:1.6;margin-bottom:8px;">
                {rec['detail']}
              </div>
              <div style="margin-top:4px;">{hours_html}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Run `python main.py --no-lstm` first to generate the energy management report, then refresh.")

        # Live fallback using only historical data
        section("Live Historical Analysis (run pipeline for forecast peaks)")
        demand_table_live = build_hourly_demand_table(hourly_f)

        fig_live = go.Figure()
        bar_colors_live = demand_table_live["demand_label"].map(
            {"Low": "#4caf50", "Medium": "#ff9800", "High": "#f44336", "Peak": "#8b0000"}
        ).tolist()
        fig_live.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in demand_table_live["hour"]],
            y=demand_table_live["avg_kw"],
            marker_color=bar_colors_live,
            text=demand_table_live["demand_label"],
            textposition="outside",
        ))
        t90 = float(np.quantile(hourly_f["Global_active_power"].dropna(), 0.90))
        fig_live.add_hline(y=t90, line_dash="dash", line_color=RED,
                           annotation_text=f"90th pct threshold = {t90:.2f} kW")
        fig_live.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            xaxis_title="Hour", yaxis_title="Avg kW",
            margin=dict(l=40, r=20, t=20, b=60),
            xaxis=dict(tickangle=45),
        )
        st.plotly_chart(fig_live, use_container_width=True)

        # Live recommendation cards from historical data only
        hourly_avg_live = hourly_f.groupby(hourly_f.index.hour)["Global_active_power"].mean()
        peak_hrs_live   = hourly_avg_live.nlargest(5).index.tolist()
        hour_strs_live  = ", ".join(f"{h:02d}:00" for h in sorted(peak_hrs_live))

        live_recs = [
            {"category": "load_shift", "icon": "SHIFT",
             "title": "Shift Flexible Loads to Off-Peak",
             "detail": f"Historically high demand at {hour_strs_live}. Move washing machine, EV charging to 01:00-06:00.",
             "hours": sorted(peak_hrs_live)},
            {"category": "hvac", "icon": "HVAC",
             "title": "Pre-condition HVAC Before Peak Windows",
             "detail": f"Sub-metering 3 (HVAC) is the biggest flexible load. Pre-cool/pre-heat 30-60 min before {hour_strs_live}.",
             "hours": sorted(peak_hrs_live)},
            {"category": "battery", "icon": "BATT",
             "title": "Pre-charge Storage During Off-Peak (01:00-06:00)",
             "detail": "Charge home battery or EV overnight. Discharge during peak windows to reduce grid import.",
             "hours": list(range(1, 7))},
            {"category": "voltage", "icon": "VOLT",
             "title": f"Monitor Voltage During Peak Hours",
             "detail": f"Target: {VOLTAGE_LOW}-{VOLTAGE_HIGH} V. Voltage sags often coincide with peak-demand periods.",
             "hours": sorted(peak_hrs_live)},
            {"category": "tariff", "icon": "COST",
             "title": "Review Time-of-Use Tariff Alignment",
             "detail": f"Peak demand clusters at {hour_strs_live}. Even a 30-min shift can reduce your monthly bill.",
             "hours": sorted(peak_hrs_live)},
        ]
        CARD_META_FB = {
            "load_shift": {"color": "#f7b731", "bg": "rgba(247,183,49,0.08)",  "border": "#f7b731"},
            "hvac":       {"color": "#4facfe", "bg": "rgba(79,172,254,0.08)",  "border": "#4facfe"},
            "battery":    {"color": "#00d97e", "bg": "rgba(0,217,126,0.08)",   "border": "#00d97e"},
            "voltage":    {"color": "#ff6b6b", "bg": "rgba(255,107,107,0.08)", "border": "#ff6b6b"},
            "tariff":     {"color": "#a78bfa", "bg": "rgba(167,139,250,0.08)", "border": "#a78bfa"},
        }
        for rec in live_recs:
            meta = CARD_META_FB[rec["category"]]
            hours_html = " ".join(
                f'<span style="background:{meta["color"]}22;color:{meta["color"]};'
                f'border-radius:4px;padding:1px 7px;font-size:0.78rem;">{h:02d}:00</span>'
                for h in rec.get("hours", [])
            )
            st.markdown(f"""
            <div style="background:{meta['bg']};border:1px solid {meta['border']};
                        border-left:4px solid {meta['border']};border-radius:10px;
                        padding:14px 18px;margin-bottom:12px;">
              <div style="font-size:0.95rem;font-weight:700;color:{meta['color']};margin-bottom:6px;">
                [{rec['icon']}] &nbsp; {rec['title']}
              </div>
              <div style="color:#c9d1e8;font-size:0.875rem;line-height:1.6;margin-bottom:8px;">
                {rec['detail']}
              </div>
              <div style="margin-top:4px;">{hours_html}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📋 Data Quality":
    st.title("📋 Data Quality Report")

    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi("Total Records", f"{qa_report['total_rows']:,}")
    with col2: kpi("Duplicate Timestamps", f"{qa_report['duplicate_timestamps']:,}",
                   "None found ✓" if qa_report['duplicate_timestamps'] == 0 else "Removed", True)
    with col3: kpi("Date Start", qa_report['date_range'][0][:10])
    with col4: kpi("Date End",   qa_report['date_range'][1][:10])

    st.markdown("---")
    section("Missing Values per Column (before imputation)")
    missing_before = pd.Series(qa_report["missing_before"]).sort_values(ascending=False)
    missing_pct    = pd.Series(qa_report["missing_pct"])
    mv_df = pd.DataFrame({
        "Missing Count": missing_before,
        "Missing %":     missing_pct,
    })
    fig_mv = px.bar(
        mv_df, x=mv_df.index, y="Missing Count",
        text=mv_df["Missing %"].map(lambda v: f"{v:.1f}%"),
        color="Missing %", color_continuous_scale="Reds",
        template=PLOTLY_TEMPLATE
    )
    fig_mv.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=50))
    st.plotly_chart(fig_mv, use_container_width=True)

    section("Missing Values After Imputation")
    missing_after = pd.Series(qa_report["missing_after"])
    after_df = pd.DataFrame({"Remaining Missing": missing_after})
    st.dataframe(after_df.style.highlight_max(color="#ff6b6b").format("{:.0f}"),
                 use_container_width=True)

    section("Dataset Statistics (Hourly)")
    st.dataframe(hourly_df.describe().T.style.background_gradient(cmap="Blues", subset=["mean","std"]),
                 use_container_width=True)

    section("Sample Raw Data (Hourly – first 100 rows)")
    st.dataframe(hourly_df.head(100), use_container_width=True)
