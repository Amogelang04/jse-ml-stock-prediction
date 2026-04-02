"""Plotting utilities for model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def plot_predictions(dates, actual, predicted, model_name: str) -> None:
    """Save a time-series plot of actual vs predicted prices."""
    plt.figure(figsize=(11, 5))
    plt.plot(dates, actual, label="Actual Close")
    plt.plot(dates, predicted, label="Predicted Close")
    plt.title(f"Actual vs Predicted Closing Price: {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.png")
    plt.close()


def save_metrics_table(metrics: list[dict]) -> pd.DataFrame:
    """Save model metrics to CSV."""
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    return metrics_df
