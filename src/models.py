"""
models.py
─────────
Step 7 – Train forecasting models.
Step 8 – Evaluate with MAE, RMSE, R² and plot actual vs predicted.

Models
──────
  • LinearRegression   – baseline
  • RandomForest       – ensemble tree model
  • XGBoost            – gradient boosting
  • LSTM               – deep-learning sequence model (optional, requires TF)

Results are persisted to outputs/models/ and outputs/plots/.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

MODEL_DIR = Path("outputs/models")
PLOT_DIR  = Path("outputs/plots")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns used by all sklearn / xgb models
CALENDAR_FEATURES = [
    "hour", "day_of_month", "month", "year", "weekday", "is_weekend",
    "quarter", "day_of_year", "week_of_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]


def _get_feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    """Return all feature columns present in df (excluding the target)."""
    lag_roll = [c for c in df.columns if "_lag" in c or "_roll_" in c]
    candidates = CALENDAR_FEATURES + lag_roll
    return [c for c in candidates if c in df.columns and c != target]


def train_test_split_ts(df: pd.DataFrame, test_ratio: float = 0.2):
    """Chronological split – no shuffling."""
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]


def evaluate(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def plot_actual_vs_predicted(
    y_true, y_pred, label: str, target: str
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Time-series overlay
    axes[0].plot(np.array(y_true), label="Actual",    color="#1f77b4", lw=1)
    axes[0].plot(np.array(y_pred), label="Predicted", color="#ff7f0e", lw=1, alpha=0.8)
    axes[0].set_title(f"{label} – Actual vs Predicted ({target})", fontsize=13)
    axes[0].set_xlabel("Test Samples")
    axes[0].set_ylabel(target)
    axes[0].legend()

    # Scatter
    axes[1].scatter(np.array(y_true), np.array(y_pred),
                    alpha=0.3, s=8, color="#2ca02c")
    lim = [min(np.array(y_true).min(), np.array(y_pred).min()),
           max(np.array(y_true).max(), np.array(y_pred).max())]
    axes[1].plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
    axes[1].set_title(f"{label} – Scatter (Actual vs Predicted)", fontsize=13)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].legend()

    fig.tight_layout()
    safe = label.replace(" ", "_").lower()
    fig.savefig(PLOT_DIR / f"pred_{safe}.png", dpi=150)
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Classical ML models
# ────────────────────────────────────────────────────────────────────────────

def train_linear_regression(
    df: pd.DataFrame, target: str = "Global_active_power"
) -> dict:
    feat_cols = _get_feature_cols(df, target)
    train, test = train_test_split_ts(df)
    X_train, y_train = train[feat_cols].values, train[target].values
    X_test,  y_test  = test[feat_cols].values,  test[target].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = evaluate(y_test, y_pred)
    fig = plot_actual_vs_predicted(y_test, y_pred, "Linear Regression", target)

    joblib.dump({"model": model, "scaler": scaler, "features": feat_cols},
                MODEL_DIR / "linear_regression.pkl")

    return {"name": "Linear Regression", "metrics": metrics,
            "fig": fig, "y_test": y_test, "y_pred": y_pred,
            "test_index": test.index}


def train_random_forest(
    df: pd.DataFrame, target: str = "Global_active_power", n_estimators: int = 200
) -> dict:
    feat_cols = _get_feature_cols(df, target)
    train, test = train_test_split_ts(df)
    X_train, y_train = train[feat_cols].values, train[target].values
    X_test,  y_test  = test[feat_cols].values,  test[target].values

    model = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=42, max_depth=16
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    fig = plot_actual_vs_predicted(y_test, y_pred, "Random Forest", target)

    joblib.dump({"model": model, "features": feat_cols},
                MODEL_DIR / "random_forest.pkl")

    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    return {"name": "Random Forest", "metrics": metrics,
            "fig": fig, "importances": importances,
            "y_test": y_test, "y_pred": y_pred,
            "test_index": test.index}


def train_xgboost(
    df: pd.DataFrame, target: str = "Global_active_power"
) -> dict:
    feat_cols = _get_feature_cols(df, target)
    train, test = train_test_split_ts(df)
    X_train, y_train = train[feat_cols].values, train[target].values
    X_test,  y_test  = test[feat_cols].values,  test[target].values

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=30,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    fig = plot_actual_vs_predicted(y_test, y_pred, "XGBoost", target)

    joblib.dump({"model": model, "features": feat_cols},
                MODEL_DIR / "xgboost.pkl")

    importances = pd.Series(
        model.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)
    return {"name": "XGBoost", "metrics": metrics,
            "fig": fig, "importances": importances,
            "y_test": y_test, "y_pred": y_pred,
            "test_index": test.index}


# ────────────────────────────────────────────────────────────────────────────
# LSTM (deep-learning)
# ────────────────────────────────────────────────────────────────────────────

def train_lstm(
    df: pd.DataFrame, target: str = "Global_active_power",
    look_back: int = 24, epochs: int = 20, batch_size: int = 64
) -> dict:
    """
    Sequence-to-one LSTM.
    Uses only the target column as the input sequence for simplicity;
    extend `feat_cols` to include engineered features if needed.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        return {"name": "LSTM", "metrics": {}, "fig": None,
                "error": "TensorFlow not installed. Run: pip install tensorflow"}

    series = df[target].values.reshape(-1, 1)
    scaler = StandardScaler()
    series_s = scaler.fit_transform(series)

    split = int(len(series_s) * 0.8)

    def make_sequences(data, n):
        X, y = [], []
        for i in range(n, len(data)):
            X.append(data[i - n:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y_all = make_sequences(series_s, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    cb = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, callbacks=[cb], verbose=0)

    y_pred_s = model.predict(X_test, verbose=0).flatten()
    y_pred   = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_true   = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate(y_true, y_pred)
    fig = plot_actual_vs_predicted(y_true, y_pred, "LSTM", target)

    model.save(str(MODEL_DIR / "lstm_model.keras"))
    joblib.dump({"scaler": scaler, "look_back": look_back},
                MODEL_DIR / "lstm_meta.pkl")

    return {"name": "LSTM", "metrics": metrics,
            "fig": fig, "y_test": y_true, "y_pred": y_pred}


# ────────────────────────────────────────────────────────────────────────────
# Comparison summary
# ────────────────────────────────────────────────────────────────────────────

def compare_models(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.get("metrics"):
            rows.append({"Model": r["name"], **r["metrics"]})
    df = pd.DataFrame(rows).set_index("Model")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics_info = [("MAE", True), ("RMSE", True), ("R2", False)]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, (metric, lower_better), color in zip(axes, metrics_info, colors):
        ax.bar(df.index, df[metric], color=color, edgecolor="white")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xticklabels(df.index, rotation=20, ha="right")
    fig.suptitle("Model Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison.png", dpi=150)

    df.to_csv(Path("outputs/reports") / "model_metrics.csv")
    return df


def plot_feature_importance(importances: pd.Series, model_name: str) -> plt.Figure:
    top = importances.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="#4c72b0")
    ax.set_title(f"{model_name} – Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    fig.savefig(PLOT_DIR / f"importance_{safe}.png", dpi=150)
    return fig
