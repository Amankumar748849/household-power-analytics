"""
main.py
───────
Orchestrates the full household-power-consumption analytics pipeline.

Usage
─────
    python main.py              # runs all steps including LSTM
    python main.py --no-lstm    # skip LSTM (faster, no TF required)
    python main.py --daily      # use daily granularity for ML models
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

# ── Project modules ──────────────────────────────────────────────────────────
from src.data_processing import run_pipeline
from src.features import build_hourly_features, build_daily_features
from src.eda import (
    plot_daily_trend,
    plot_hourly_profile,
    plot_weekday_weekend,
    plot_monthly_seasonality,
    plot_correlation,
    plot_submetering,
    plot_distribution,
    plot_yoy_comparison,
)
from src.models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_lstm,
    compare_models,
    plot_feature_importance,
)
from src.peak_analysis import (
    detect_historical_peaks,
    build_hourly_demand_table,
    forecast_peaks,
    build_forecast_hour_summary,
    energy_management_report,
    plot_peak_heatmap,
    plot_forecast_with_peaks,
    plot_voltage_band,
    plot_hourly_demand_bar,
)

REPORTS_DIR = Path("outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")


def main(use_lstm: bool = True, use_daily: bool = False):
    t0 = time.time()

    # ── Step 1-4: Data loading, cleaning, aggregation ────────────────────────
    banner("STEP 1-4 | Data Loading & Preprocessing")
    data = run_pipeline()
    minute_df = data["minute"]
    hourly_df  = data["hourly"]
    daily_df   = data["daily"]
    report     = data["report"]

    print(f"  [OK] Loaded {report['total_rows']:,} minute-level records")
    print(f"  [OK] Date range: {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"  [OK] Missing (before): {sum(report['missing_before'].values())}")
    print(f"  [OK] Duplicate timestamps removed: {report['duplicate_timestamps']}")
    print(f"  [OK] Hourly rows: {len(hourly_df):,} | Daily rows: {len(daily_df):,}")

    # Save QA report
    with open(REPORTS_DIR / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Step 5: Feature engineering ──────────────────────────────────────────
    banner("STEP 5 | Feature Engineering")
    if use_daily:
        feat_df = build_daily_features(daily_df)
        target  = "Energy_kWh"
        gran    = "daily"
    else:
        feat_df = build_hourly_features(hourly_df)
        target  = "Global_active_power"
        gran    = "hourly"
    print(f"  [OK] Feature matrix ({gran}): {feat_df.shape[0]:,} rows x {feat_df.shape[1]} cols")

    # ── Step 6: EDA ──────────────────────────────────────────────────────────
    banner("STEP 6 | Exploratory Data Analysis")
    plot_daily_trend(daily_df);          print("  [OK] daily_trend.png")
    plot_hourly_profile(hourly_df);      print("  [OK] hourly_profile.png")
    plot_weekday_weekend(hourly_df);     print("  [OK] weekday_weekend.png")
    plot_monthly_seasonality(daily_df);  print("  [OK] monthly_seasonality.png")
    plot_correlation(hourly_df, "Hourly Feature Correlation"); print("  [OK] correlation_heatmap.png")
    plot_submetering(daily_df);          print("  [OK] submetering.png")
    plot_distribution(hourly_df);        print("  [OK] distribution.png")
    plot_yoy_comparison(daily_df);       print("  [OK] yoy_comparison.png")

    # ── Step 7-8: Model training & evaluation ────────────────────────────────
    banner("STEP 7-8 | Model Training & Evaluation")
    results = []

    print("  --> Training Linear Regression ...")
    lr_res = train_linear_regression(feat_df, target=target)
    results.append(lr_res)
    print(f"     MAE={lr_res['metrics']['MAE']}  RMSE={lr_res['metrics']['RMSE']}  R2={lr_res['metrics']['R2']}")

    print("  --> Training Random Forest ...")
    rf_res = train_random_forest(feat_df, target=target)
    results.append(rf_res)
    print(f"     MAE={rf_res['metrics']['MAE']}  RMSE={rf_res['metrics']['RMSE']}  R2={rf_res['metrics']['R2']}")
    plot_feature_importance(rf_res["importances"], "Random Forest")

    print("  --> Training XGBoost ...")
    xgb_res = train_xgboost(feat_df, target=target)
    results.append(xgb_res)
    print(f"     MAE={xgb_res['metrics']['MAE']}  RMSE={xgb_res['metrics']['RMSE']}  R2={xgb_res['metrics']['R2']}")
    plot_feature_importance(xgb_res["importances"], "XGBoost")

    if use_lstm:
        print("  --> Training LSTM (this may take a few minutes) ...")
        lstm_res = train_lstm(
            hourly_df if not use_daily else daily_df,
            target=target, look_back=24, epochs=20
        )
        if lstm_res.get("error"):
            print(f"     [!] LSTM skipped: {lstm_res['error']}")
        else:
            results.append(lstm_res)
            print(f"     MAE={lstm_res['metrics']['MAE']}  RMSE={lstm_res['metrics']['RMSE']}  R2={lstm_res['metrics']['R2']}")

    metrics_df = compare_models(results)
    print("\n  -- Model Comparison --")
    print(metrics_df.to_string())

    # ── Step 9: Peak analysis & energy management ─────────────────────────────
    banner("STEP 9 | Peak Analysis & Energy Management")
    best = max(results, key=lambda r: r["metrics"].get("R2", -999))
    print(f"  [OK] Best model by R2: {best['name']} (R2={best['metrics']['R2']})")

    # -- Demand table (historical)
    demand_table = build_hourly_demand_table(hourly_df)
    print("  [OK] Hourly demand table built")

    # -- Forecast peaks
    test_index   = best.get("test_index", hourly_df.index[-len(best["y_pred"]):])
    forecast_df, threshold = forecast_peaks(test_index, best["y_pred"])
    print(f"  [OK] Peak threshold (90th pct): {threshold:.4f} kW")

    fc_hour_summary = build_forecast_hour_summary(forecast_df, threshold)

    # -- Management report
    mgmt = energy_management_report(
        hourly_df, forecast_df, threshold,
        demand_table=demand_table,
        fc_hour_summary=fc_hour_summary,
    )
    with open(REPORTS_DIR / "energy_management.json", "w") as f:
        json.dump(mgmt, f, indent=2)

    # -- Plots
    plot_peak_heatmap(hourly_df);                              print("  [OK] peak_heatmap.png")
    plot_hourly_demand_bar(demand_table);                      print("  [OK] hourly_demand_bar.png")
    plot_forecast_with_peaks(forecast_df, threshold,
                             model_name=best["name"]);         print("  [OK] forecast_peaks.png")
    plot_voltage_band(hourly_df);                              print("  [OK] voltage_band.png")

    print("\n  -- Energy Management Recommendations --")
    for rec in mgmt["recommendations"]:
        print(f"  [{rec['icon']}] {rec['title']}")
        print(f"       {rec['detail'][:120]}...")

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  Outputs saved to: outputs/plots/  outputs/models/  outputs/reports/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Household Power Consumption Pipeline")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--daily",   action="store_true", help="Use daily granularity for ML")
    args = parser.parse_args()
    main(use_lstm=not args.no_lstm, use_daily=args.daily)
