<<<<<<< HEAD
# Household Power Consumption Analytics

A complete end-to-end Python data-analytics and forecasting project for the
**UCI Household Electric Power Consumption** dataset.

---

## Project Structure

```
.
├── household_power_consumption.csv   # Raw dataset
├── main.py                           # Full pipeline (CLI entry-point)
├── requirements.txt
├── src/
│   ├── data_processing.py            # Steps 1-4: Load, clean, aggregate
│   ├── features.py                   # Step  5: Feature engineering
│   ├── eda.py                        # Step  6: Exploratory analysis & plots
│   ├── models.py                     # Steps 7-8: ML/DL training & evaluation
│   └── peak_analysis.py              # Step  9: Peak detection & recommendations
├── dashboard/
│   └── app.py                        # Streamlit interactive dashboard
└── outputs/
    ├── plots/                        # Generated PNG charts
    ├── models/                       # Saved model files (.pkl / .keras)
    └── reports/                      # JSON & CSV reports
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full analytics pipeline
```bash
# Without LSTM (fast, ~20 seconds)
python main.py --no-lstm

# With LSTM deep-learning model (requires TensorFlow, ~5 min)
python main.py

# Use daily granularity instead of hourly
python main.py --no-lstm --daily
```

### 3. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```
Then open `http://localhost:8501` in your browser.

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_processing.py` | Load CSV and parse datetime index |
| 2 | `data_processing.py` | Combine Date + Time into proper DatetimeIndex |
| 3 | `data_processing.py` | Handle missing values, duplicates, invalid measurements |
| 4 | `data_processing.py` | Aggregate to hourly and daily granularities |
| 5 | `features.py` | Calendar, cyclical, lag, and rolling features |
| 6 | `eda.py` | Daily trends, hourly peaks, weekday/weekend, seasonality |
| 7 | `models.py` | Train Linear Regression, Random Forest, XGBoost, LSTM |
| 8 | `models.py` | Evaluate MAE / RMSE / R² and plot actual vs predicted |
| 9 | `peak_analysis.py` | Identify peak periods and energy-management recommendations |

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Baseline | Scaled features |
| Random Forest | Ensemble | 200 trees, max_depth=16 |
| XGBoost | Gradient Boosting | 500 estimators, early stopping |
| LSTM | Deep Learning | 2-layer LSTM, look-back=24h |

---

## Dashboard Pages

| Page | Contents |
|------|----------|
| Overview | KPI cards, daily trend, sub-metering donut, monthly bar |
| Trends & EDA | Hourly profile, weekday/weekend, distribution, correlation |
| Deep Dive | Year-over-year, heatmap, sub-metering trend, voltage scatter |
| Model Results | Metric comparison bars, actual vs predicted plots, importances |
| Peak Analysis | Consumption heatmap, top-10 peaks, forecast overlay, recommendations |
| Data Quality | Missing values, imputation stats, descriptive statistics |

---

## Dataset

**UCI Machine Learning Repository** – Individual household electric power consumption.
- ~2 million one-minute readings (Dec 2006 – Nov 2010)
- Columns: `Global_active_power`, `Global_reactive_power`, `Voltage`,
  `Global_intensity`, `Sub_metering_1/2/3`
=======
# household-power-analytics
>>>>>>> 0cee9f2583fdeef43761aa17e9e2dd4e9d55566c
