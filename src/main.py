from pathlib import Path

from analytics import (
    load_data,
    dataset_info,
    descriptive_statistics,
    closing_price_statistics,
    daily_returns,
    moving_averages,
    volatility_analysis,
    drawdown_analysis,
    monthly_performance,
    yearly_performance,
    correlation_analysis,
    risk_metrics,
    extreme_days,
)

from prediction import (
    prepare_data,
    train_models,
    evaluate_models,
    prediction_results,
)

from visualizations import (
    plot_price_and_moving_averages,
    plot_daily_returns,
    plot_rolling_volatility,
    plot_drawdown,
    plot_monthly_performance,
    plot_yearly_performance,
    plot_correlation,
    plot_predictions,
)


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "banknifty.csv"

IMAGE_DIR = ROOT / "images"

REPORT_DIR = ROOT / "report"

IMAGE_DIR.mkdir(
    exist_ok=True
)

REPORT_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# PROJECT START
# ==================================================

print("\n" + "=" * 70)

print(
    "BANK NIFTY MARKET ANALYTICS & FORECASTING"
)

print("=" * 70)


# ==================================================
# LOAD DATA
# ==================================================

df = load_data(
    DATA_PATH
)


# ==================================================
# DATASET OVERVIEW
# ==================================================

dataset_info(
    df
)


# ==================================================
# DESCRIPTIVE ANALYSIS
# ==================================================

descriptive_statistics(
    df
)

closing_price_statistics(
    df
)


# ==================================================
# DAILY RETURNS
# ==================================================

df = daily_returns(
    df
)


# ==================================================
# MOVING AVERAGES
# ==================================================

df = moving_averages(
    df
)


# ==================================================
# VOLATILITY
# ==================================================

df = volatility_analysis(
    df
)


# ==================================================
# DRAWDOWN
# ==================================================

df = drawdown_analysis(
    df
)


# ==================================================
# MONTHLY PERFORMANCE
# ==================================================

monthly = monthly_performance(
    df
)


# ==================================================
# YEARLY PERFORMANCE
# ==================================================

yearly = yearly_performance(
    df
)


# ==================================================
# CORRELATION
# ==================================================

corr = correlation_analysis(
    df
)


# ==================================================
# RISK METRICS
# ==================================================

metrics = risk_metrics(
    df
)


# ==================================================
# EXTREME TRADING DAYS
# ==================================================

best_days, worst_days = extreme_days(
    df
)


print(
    "\n========== BEST 10 TRADING DAYS ==========\n"
)

print(
    best_days.to_string(
        index=False
    )
)


print(
    "\n========== WORST 10 TRADING DAYS ==========\n"
)

print(
    worst_days.to_string(
        index=False
    )
)


# ==================================================
# VISUALIZATIONS
# ==================================================

print(
    "\n========== GENERATING VISUALIZATIONS ==========\n"
)


plot_price_and_moving_averages(
    df,
    IMAGE_DIR
)


plot_daily_returns(
    df,
    IMAGE_DIR
)


plot_rolling_volatility(
    df,
    IMAGE_DIR
)


plot_drawdown(
    df,
    IMAGE_DIR
)


plot_monthly_performance(
    monthly,
    IMAGE_DIR
)


plot_yearly_performance(
    yearly,
    IMAGE_DIR
)


plot_correlation(
    corr,
    IMAGE_DIR
)


# ==================================================
# SMALL ML EXPERIMENT
# ==================================================

print(
    "\n" + "=" * 70
)

print(
    "SMALL ML EXPERIMENT: NEXT-DAY FORECASTING"
)

print("=" * 70)


X_train, X_test, y_train, y_test, dates = (
    prepare_data(
        df
    )
)


# Train SVR models

models = train_models(
    X_train,
    y_train
)


# Evaluate models

comparison = evaluate_models(
    models,
    X_test,
    y_test
)


print(
    "\n========== MODEL COMPARISON ==========\n"
)

print(
    comparison.to_string(
        index=False
    )
)


# ==================================================
# PREDICTION DATA
# ==================================================

predictions = prediction_results(
    models,
    X_test,
    y_test,
    dates
)


# ==================================================
# SAVE RESULTS
# ==================================================

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)


monthly.to_csv(
    REPORT_DIR / "monthly_performance.csv"
)


yearly.to_csv(
    REPORT_DIR / "yearly_performance.csv"
)


predictions.to_csv(
    REPORT_DIR / "predictions.csv",
    index=False
)


# ==================================================
# PREDICTION VISUALIZATION
# ==================================================

plot_predictions(
    predictions,
    IMAGE_DIR
)


# ==================================================
# FINAL MESSAGE
# ==================================================

print(
    "\n" + "=" * 70
)

print(
    "ANALYSIS COMPLETE"
)

print("=" * 70)

print(
    f"\nCharts saved to:\n{IMAGE_DIR}"
)

print(
    f"\nResults saved to:\n{REPORT_DIR}"
)