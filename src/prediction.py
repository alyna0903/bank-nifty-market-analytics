import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# PREPARE DATA
# ==================================================

def prepare_data(df):
    """
    Creates lag-based features for next-day
    Bank Nifty closing-price prediction.

    Features:
    - Previous day's closing price
    - 5-day moving average
    - 20-day moving average

    Target:
    - Current day's closing price
    """

    data = df.copy()

    # Previous day's close
    data["Previous Close"] = data["close"].shift(1)

    # Moving averages shifted by one day so that
    # only information available before the prediction
    # is used.
    data["MA5 Previous"] = (
        data["close"]
        .rolling(window=5)
        .mean()
        .shift(1)
    )

    data["MA20 Previous"] = (
        data["close"]
        .rolling(window=20)
        .mean()
        .shift(1)
    )

    data = data.dropna().reset_index(drop=True)

    features = [
        "Previous Close",
        "MA5 Previous",
        "MA20 Previous"
    ]

    X = data[features]
    y = data["close"]

    # Chronological 80/20 split
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = data["datetime"].iloc[
        split_index:
    ].reset_index(drop=True)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates
    )


# ==================================================
# TRAIN SVR MODELS
# ==================================================

def train_models(X_train, y_train):
    """
    Train three simple SVR models.
    """

    linear = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svr",
            SVR(
                kernel="linear",
                C=1000
            )
        )
    ])

    polynomial = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svr",
            SVR(
                kernel="poly",
                degree=2,
                C=100
            )
        )
    ])

    rbf = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=1000,
                gamma="scale"
            )
        )
    ])

    linear.fit(
        X_train,
        y_train
    )

    polynomial.fit(
        X_train,
        y_train
    )

    rbf.fit(
        X_train,
        y_train
    )

    return {
        "Linear SVR": linear,
        "Polynomial SVR": polynomial,
        "RBF SVR": rbf
    }


# ==================================================
# NAIVE BASELINE
# ==================================================

def naive_baseline(X_test, y_test):
    """
    Naive forecasting baseline.

    Predict today's closing price using
    the previous day's closing price.

    This gives us a simple benchmark against
    which the SVR models can be evaluated.
    """

    predictions = X_test[
        "Previous Close"
    ].values

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        predictions,
        mae,
        rmse,
        r2
    )


# ==================================================
# EVALUATE ONE MODEL
# ==================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate one model.
    """

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        predictions,
        mae,
        rmse,
        r2
    )


# ==================================================
# EVALUATE ALL MODELS
# ==================================================

def evaluate_models(
    models,
    X_test,
    y_test
):
    """
    Compare the naive baseline with
    the three SVR models.
    """

    results = []

    # ------------------------------
    # Naive baseline
    # ------------------------------

    (
        baseline_predictions,
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ) = naive_baseline(
        X_test,
        y_test
    )

    results.append({
        "model": "Naive Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "R2": baseline_r2
    })

    # ------------------------------
    # SVR models
    # ------------------------------

    for name, model in models.items():

        (
            predictions,
            mae,
            rmse,
            r2
        ) = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by="RMSE"
        )
        .reset_index(drop=True)
    )

    return results_df


# ==================================================
# PREDICTION RESULTS
# ==================================================

def prediction_results(
    models,
    X_test,
    y_test,
    dates
):
    """
    Creates a DataFrame containing
    actual values and model predictions.
    """

    predictions_df = pd.DataFrame({
        "datetime": dates,
        "actual": y_test.values
    })

    # Naive baseline
    predictions_df[
        "Naive Baseline"
    ] = X_test[
        "Previous Close"
    ].values

    # SVR predictions
    for name, model in models.items():

        predictions_df[name] = (
            model.predict(
                X_test
            )
        )

    return predictions_df