import matplotlib.pyplot as plt
from pathlib import Path


# ==================================================
# HELPER FUNCTION
# ==================================================

def save_plot(fig, output_dir, filename):
    """
    Saves a matplotlib figure to the images folder.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = output_dir / filename

    fig.tight_layout()

    fig.savefig(
        file_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {file_path}")


# ==================================================
# PRICE + MOVING AVERAGES
# ==================================================

def plot_price_and_moving_averages(
    df,
    output_dir
):
    """
    Plot Bank Nifty closing price along with
    20, 50, 100 and 200-day moving averages.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        df["datetime"],
        df["close"],
        label="Closing Price"
    )

    ax.plot(
        df["datetime"],
        df["MA20"],
        label="MA20"
    )

    ax.plot(
        df["datetime"],
        df["MA50"],
        label="MA50"
    )

    ax.plot(
        df["datetime"],
        df["MA100"],
        label="MA100"
    )

    ax.plot(
        df["datetime"],
        df["MA200"],
        label="MA200"
    )

    ax.set_title(
        "Bank Nifty Closing Price and Moving Averages"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Bank Nifty Index Level"
    )

    ax.legend()

    save_plot(
        fig,
        output_dir,
        "01_price_moving_averages.png"
    )


# ==================================================
# DAILY RETURNS
# ==================================================

def plot_daily_returns(
    df,
    output_dir
):
    """
    Plot daily percentage returns.
    """

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["datetime"],
        df["Daily Return (%)"],
        linewidth=0.8
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        "Bank Nifty Daily Returns"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Daily Return (%)"
    )

    save_plot(
        fig,
        output_dir,
        "02_daily_returns.png"
    )


# ==================================================
# ROLLING VOLATILITY
# ==================================================

def plot_rolling_volatility(
    df,
    output_dir
):
    """
    Plot 30-day rolling annualized volatility.
    """

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["datetime"],
        df["Rolling Volatility (%)"]
    )

    ax.set_title(
        "30-Day Rolling Annualized Volatility"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Volatility (%)"
    )

    save_plot(
        fig,
        output_dir,
        "03_rolling_volatility.png"
    )


# ==================================================
# DRAWDOWN
# ==================================================

def plot_drawdown(
    df,
    output_dir
):
    """
    Plot Bank Nifty percentage drawdown
    from its running peak.
    """

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["datetime"],
        df["Drawdown (%)"]
    )

    ax.fill_between(
        df["datetime"],
        df["Drawdown (%)"],
        0,
        alpha=0.3
    )

    ax.set_title(
        "Bank Nifty Drawdown"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Drawdown (%)"
    )

    save_plot(
        fig,
        output_dir,
        "04_drawdown.png"
    )


# ==================================================
# MONTHLY PERFORMANCE
# ==================================================

def plot_monthly_performance(
    monthly,
    output_dir
):
    """
    Plot average monthly closing prices.
    """

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    monthly.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Bank Nifty Closing Price by Month"
    )

    ax.set_xlabel("Month")

    ax.set_ylabel(
        "Average Closing Price"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    save_plot(
        fig,
        output_dir,
        "05_monthly_average_close.png"
    )


# ==================================================
# YEARLY PERFORMANCE
# ==================================================

def plot_yearly_performance(
    yearly,
    output_dir
):
    """
    Plot average yearly closing prices.
    """

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    yearly.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Average Bank Nifty Closing Price by Year"
    )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Average Closing Price"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    save_plot(
        fig,
        output_dir,
        "06_yearly_average_close.png"
    )


# ==================================================
# CORRELATION HEATMAP
# ==================================================

def plot_correlation(
    correlation,
    output_dir
):
    """
    Plot correlation matrix for OHLC variables.
    """

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        correlation.values,
        interpolation="nearest"
    )

    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.index))
    )

    ax.set_xticklabels(
        correlation.columns
    )

    ax.set_yticklabels(
        correlation.index
    )

    ax.set_title(
        "Bank Nifty OHLC Correlation Matrix"
    )

    # Add correlation values
    for i in range(
        len(correlation.index)
    ):

        for j in range(
            len(correlation.columns)
        ):

            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    save_plot(
        fig,
        output_dir,
        "07_correlation_heatmap.png"
    )


# ==================================================
# SVR PREDICTIONS
# ==================================================

def plot_predictions(
    prediction_df,
    output_dir
):
    """
    Plot actual values against the naive baseline
    and SVR predictions.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        prediction_df["datetime"],
        prediction_df["actual"],
        label="Actual",
        linewidth=2
    )

    if "Naive Baseline" in prediction_df.columns:

        ax.plot(
            prediction_df["datetime"],
            prediction_df["Naive Baseline"],
            label="Naive Baseline",
            alpha=0.8
        )

    for column in [
        "Linear SVR",
        "Polynomial SVR",
        "RBF SVR"
    ]:

        if column in prediction_df.columns:

            ax.plot(
                prediction_df["datetime"],
                prediction_df[column],
                label=column
            )

    ax.set_title(
        "Next-Day Bank Nifty Predictions vs Actual Values"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Bank Nifty Closing Price"
    )

    ax.legend()

    save_plot(
        fig,
        output_dir,
        "08_predictions_vs_actual.png"
    )