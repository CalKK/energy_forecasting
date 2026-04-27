from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    actual: pd.Series,
    forecast: pd.DataFrame | pd.Series,
    output_path: str | Path,
    title: str = "Energy Forecast",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    actual.plot(ax=ax, label="Actual")

    if isinstance(forecast, pd.DataFrame):
        forecast["forecast"].plot(ax=ax, label="Forecast")
        if {"lower", "upper"}.issubset(forecast.columns):
            ax.fill_between(
                forecast.index,
                forecast["lower"].astype(float).values,
                forecast["upper"].astype(float).values,
                alpha=0.15,
                label="Confidence interval",
            )
    else:
        forecast.plot(ax=ax, label="Forecast")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kilowatt-hours")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
