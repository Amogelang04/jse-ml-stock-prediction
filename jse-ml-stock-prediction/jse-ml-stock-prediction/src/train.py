"""Train and evaluate baseline ML models on JSE stock data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import download_stock_data
from features import FEATURE_COLUMNS, create_features
from plotting import plot_predictions, save_metrics_table


TICKER = "SBK.JO"
START_DATE = "2018-01-01"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    """Split data without shuffling to preserve time order."""
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def directional_accuracy(actual_close: np.ndarray, predicted_close: np.ndarray, previous_close: np.ndarray) -> float:
    """Measure how often the model gets the price direction correct."""
    actual_direction = np.sign(actual_close - previous_close)
    predicted_direction = np.sign(predicted_close - previous_close)
    return float(np.mean(actual_direction == predicted_direction))


def evaluate_model(model, X_train, y_train, X_test, y_test, test_dates, previous_close, model_name: str):
    """Fit, predict, evaluate, and save plot for one model."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    dir_acc = directional_accuracy(y_test.to_numpy(), predictions, previous_close.to_numpy())

    plot_predictions(test_dates, y_test, predictions, model_name)

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Directional_Accuracy": round(dir_acc, 4),
    }


def main() -> None:
    print(f"Downloading data for {TICKER}...")
    raw_df = download_stock_data(TICKER, start=START_DATE)

    print("Creating features...")
    model_df = create_features(raw_df)

    train_df, test_df = train_test_split_time_series(model_df, test_size=TEST_SIZE)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_next_close"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_next_close"]
    test_dates = pd.to_datetime(test_df["Date"])
    previous_close = test_df["Close"]

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            random_state=RANDOM_STATE,
        ),
    }

    metrics = []

    print("Training models...")
    for model_name, model in models.items():
        result = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            test_dates,
            previous_close,
            model_name,
        )
        metrics.append(result)
        print(f"Finished: {model_name}")
        print(result)

    metrics_df = save_metrics_table(metrics)

    print("\nModel comparison:")
    print(metrics_df)
    print("\nSaved plots and metrics to the results folder.")


if __name__ == "__main__":
    main()
