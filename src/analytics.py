import pandas as pd
import numpy as np


# ==================================================
# LOAD DATA
# ==================================================

def load_data(file_path):
    """
    Loads the dataset and performs basic cleaning.
    """

    df = pd.read_csv(file_path)

    # Convert datetime column
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Sort data by date
    df = df.sort_values(by="datetime")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ==================================================
# DATASET INFORMATION
# ==================================================

def dataset_info(df):
    """
    Prints basic dataset information.
    """

    print("\n========== DATASET INFORMATION ==========\n")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nFirst Five Rows")
    print(df.head())

    print("\nLast Five Rows")
    print(df.tail())


# ==================================================
# DESCRIPTIVE STATISTICS
# ==================================================

def descriptive_statistics(df):
    """
    Displays statistical summary of the numerical columns.
    """

    print("\n========== DESCRIPTIVE STATISTICS ==========\n")

    print(df.describe())


# ==================================================
# CLOSING PRICE STATISTICS
# ==================================================

def closing_price_statistics(df):
    """
    Calculates important statistics for the closing price.
    """

    print("\n========== CLOSING PRICE ANALYSIS ==========\n")

    close = df["close"]

    print(f"Average Closing Price      : {close.mean():.2f}")
    print(f"Median Closing Price       : {close.median():.2f}")
    print(f"Maximum Closing Price      : {close.max():.2f}")
    print(f"Minimum Closing Price      : {close.min():.2f}")
    print(f"Standard Deviation         : {close.std():.2f}")
    print(f"Variance                   : {close.var():.2f}")


# ==================================================
# DAILY RETURNS
# ==================================================

def daily_returns(df):
    """
    Calculates daily percentage returns.
    """

    df["Daily Return (%)"] = df["close"].pct_change() * 100

    print("\n========== DAILY RETURN ANALYSIS ==========\n")

    print(df["Daily Return (%)"].describe())

    return df


# ==================================================
# MOVING AVERAGES
# ==================================================

def moving_averages(df):
    """
    Calculates 20, 50, 100 and 200-day moving averages.
    """

    df["MA20"] = df["close"].rolling(window=20).mean()

    df["MA50"] = df["close"].rolling(window=50).mean()

    df["MA100"] = df["close"].rolling(window=100).mean()

    df["MA200"] = df["close"].rolling(window=200).mean()

    print("\nMoving averages calculated successfully.")

    return df


# ==================================================
# VOLATILITY ANALYSIS
# ==================================================

def volatility_analysis(df, window=30):
    """
    Calculates rolling annualized volatility using
    a 30-day rolling standard deviation of daily returns.
    """

    # Make sure daily returns exist
    if "Daily Return (%)" not in df.columns:
        df = daily_returns(df)

    # Convert percentage returns to decimal returns
    returns = df["Daily Return (%)"] / 100

    df["Rolling Volatility (%)"] = (
        returns.rolling(window=window).std()
        * np.sqrt(252)
        * 100
    )

    print("\n30-day rolling volatility calculated successfully.")

    return df


# ==================================================
# DRAWDOWN ANALYSIS
# ==================================================

def drawdown_analysis(df):
    """
    Calculates running peak and percentage drawdown.
    """

    running_peak = df["close"].cummax()

    df["Running Peak"] = running_peak

    df["Drawdown (%)"] = (
        (df["close"] - running_peak)
        / running_peak
    ) * 100

    print("\nDrawdown analysis completed.")

    return df


# ==================================================
# MONTHLY PERFORMANCE
# ==================================================

def monthly_performance(df):
    """
    Calculates average monthly closing prices.
    """

    df["Month"] = df["datetime"].dt.month_name()

    monthly = (
        df.groupby("Month")["close"]
        .mean()
        .reindex([
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ])
    )

    print("\n========== MONTHLY AVERAGE CLOSE ==========\n")

    print(monthly)

    return monthly


# ==================================================
# YEARLY PERFORMANCE
# ==================================================

def yearly_performance(df):
    """
    Calculates yearly average closing prices.
    """

    df["Year"] = df["datetime"].dt.year

    yearly = df.groupby("Year")["close"].mean()

    print("\n========== YEARLY AVERAGE CLOSE ==========\n")

    print(yearly)

    return yearly


# ==================================================
# CORRELATION ANALYSIS
# ==================================================

def correlation_analysis(df):
    """
    Correlation matrix of stock features.
    """

    correlation = df[
        ["open", "high", "low", "close"]
    ].corr()

    print("\n========== CORRELATION MATRIX ==========\n")

    print(correlation)

    return correlation


# ==================================================
# RISK & RETURN METRICS
# ==================================================

def risk_metrics(df):
    """
    Calculates major market risk and return metrics.
    """

    if "Daily Return (%)" not in df.columns:
        df = daily_returns(df)

    if "Drawdown (%)" not in df.columns:
        df = drawdown_analysis(df)

    returns = (
        df["Daily Return (%)"]
        .dropna() / 100
    )

    # Total return
    cumulative_price_return = (
        df["close"].iloc[-1]
        / df["close"].iloc[0]
    ) - 1

    # Number of years represented by dataset
    years = (
        df["datetime"].iloc[-1]
        - df["datetime"].iloc[0]
    ).days / 365.25

    # Annualized return
    annualized_return = (
        (1 + cumulative_price_return) ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    # Annualized volatility
    annualized_volatility = (
        returns.std() * np.sqrt(252)
    )

    # Sharpe ratio
    # Assumes zero risk-free rate
    sharpe_ratio = (
        annualized_return
        / annualized_volatility
        if annualized_volatility != 0
        else np.nan
    )

     # Downside deviation relative to 0% target
    downside_returns = np.minimum(returns, 0)

    downside_deviation = ( 
        np.sqrt(np.mean(downside_returns ** 2))
        *np.sqrt(252) 
)

# Sortino ratio
    sortino_ratio = (
        annualized_return / downside_deviation
        if downside_deviation != 0
        else np.nan
)

    # Maximum drawdown
    max_drawdown = df["Drawdown (%)"].min() / 100

    # Best and worst trading days
    best_day = returns.max()
    worst_day = returns.min()

    # Percentage of positive trading days
    positive_days_pct = (
        returns > 0
    ).mean()

    metrics = {
        "cumulative_price_return": cumulative_price_return,
        "annualized_price_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "best_day": best_day,
        "worst_day": worst_day,
        "positive_days_pct": positive_days_pct
    }

    print("\n========== RISK & RETURN METRICS ==========\n")

    print(
        f"Cumulative Price Return  : "
        f"{cumulative_price_return:.2%}"
    )

    print(
        f"Annualized Price Return  : "
        f"{annualized_return:.2%}"
    )

    print(
        f"Annualized Volatility    : "
        f"{annualized_volatility:.2%}"
    )

    print(
        f"Sharpe Ratio             : "
        f"{sharpe_ratio:.3f}"
    )

    print(
        f"Sortino Ratio            : "
        f"{sortino_ratio:.3f}"
    )

    print(
        f"Maximum Drawdown         : "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Best Trading Day         : "
        f"{best_day:.2%}"
    )

    print(
        f"Worst Trading Day        : "
        f"{worst_day:.2%}"
    )

    print(
        f"Positive Trading Days    : "
        f"{positive_days_pct:.2%}"
    )

    return metrics


# ==================================================
# BEST & WORST TRADING DAYS
# ==================================================

def extreme_days(df, n=10):
    """
    Returns the best and worst trading days
    based on daily percentage returns.
    """

    if "Daily Return (%)" not in df.columns:
        df = daily_returns(df)

    result = df[
        ["datetime", "close", "Daily Return (%)"]
    ].dropna()

    best_days = result.nlargest(
        n,
        "Daily Return (%)"
    )

    worst_days = result.nsmallest(
        n,
        "Daily Return (%)"
    )

    return best_days, worst_days