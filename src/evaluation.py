from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    if len(actual) == 0:
        raise ValueError("No finite values available for metric calculation.")
    error = actual - pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    denom = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
    mape = np.nanmean(np.abs(error) / denom) * 100
    smape = np.nanmean(2 * np.abs(error) / np.maximum(np.abs(actual) + np.abs(pred), 1e-9)) * 100
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "smape_percent": float(smape),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    order = ["model", "mae", "rmse", "mape_percent", "smape_percent"]
    existing = [col for col in order if col in table.columns]
    return table[existing + [col for col in table.columns if col not in existing]]
