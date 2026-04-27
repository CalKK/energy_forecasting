from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesData:
    frame: pd.DataFrame
    target_col: str
    datetime_col: str = "timestamp"

    @property
    def y(self) -> pd.Series:
        return self.frame.set_index(self.datetime_col)[self.target_col].asfreq("h")


def aggregate_to_hourly(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    frequency: str = "h",
    target_agg: str = "sum",
    numeric_agg: str = "mean",
) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    exog_cols = [col for col in numeric_cols if col != target_col]
    agg_map: dict[str, str] = {target_col: target_agg}
    agg_map.update({col: numeric_agg for col in exog_cols})

    hourly = (
        df.set_index(datetime_col)
        .sort_index()
        .resample(frequency)
        .agg(agg_map)
        .reset_index()
        .rename(columns={datetime_col: "timestamp"})
    )
    return hourly


def complete_hourly_index(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
    target_fill: str = "zero",
    exog_fill: str = "interpolate",
) -> pd.DataFrame:
    indexed = df.copy()
    indexed[timestamp_col] = pd.to_datetime(indexed[timestamp_col])
    indexed = indexed.set_index(timestamp_col).sort_index()
    full_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq="h")
    indexed = indexed.reindex(full_index)
    indexed.index.name = timestamp_col

    if target_fill == "zero":
        indexed[target_col] = indexed[target_col].fillna(0)
    elif target_fill == "interpolate":
        indexed[target_col] = indexed[target_col].interpolate(limit_direction="both")
    elif target_fill == "ffill":
        indexed[target_col] = indexed[target_col].ffill().bfill()
    else:
        raise ValueError(f"Unsupported target fill method: {target_fill}")

    exog_cols = [col for col in indexed.columns if col != target_col]
    if exog_cols:
        if exog_fill == "interpolate":
            indexed[exog_cols] = indexed[exog_cols].interpolate(limit_direction="both").ffill().bfill()
        elif exog_fill == "ffill":
            indexed[exog_cols] = indexed[exog_cols].ffill().bfill()
        elif exog_fill == "zero":
            indexed[exog_cols] = indexed[exog_cols].fillna(0)
        else:
            raise ValueError(f"Unsupported exogenous fill method: {exog_fill}")

    return indexed.reset_index()


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp", include_cyclical: bool = True) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    if include_cyclical:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp")
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    for window in rolling_windows:
        shifted = out[target_col].shift(1)
        out[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window, min_periods=max(2, window // 4)).mean()
        out[f"{target_col}_roll_std_{window}"] = shifted.rolling(window, min_periods=max(2, window // 4)).std()
    return out


def build_supervised_frame(
    df: pd.DataFrame,
    target_col: str,
    exogenous_cols: Iterable[str] | None = None,
    lags: Iterable[int] = (1, 24, 168),
    rolling_windows: Iterable[int] = (24, 168),
) -> tuple[pd.DataFrame, list[str]]:
    out = add_calendar_features(df)
    out = add_lag_features(out, target_col, lags=lags, rolling_windows=rolling_windows)

    feature_cols = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    if exogenous_cols:
        feature_cols.extend([col for col in exogenous_cols if col in out.columns])
    feature_cols.extend([col for col in out.columns if col.startswith(f"{target_col}_lag_")])
    feature_cols.extend([col for col in out.columns if col.startswith(f"{target_col}_roll_")])

    model_frame = out.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return model_frame, feature_cols


def time_train_test_split(df: pd.DataFrame, timestamp_col: str, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(df[timestamp_col]).max() - pd.Timedelta(days=test_days)
    train = df[pd.to_datetime(df[timestamp_col]) <= cutoff].copy()
    test = df[pd.to_datetime(df[timestamp_col]) > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset. Reduce test_days or inspect timestamps.")
    return train, test


def future_calendar_index(last_timestamp: pd.Timestamp, periods: int, freq: str = "h") -> pd.DataFrame:
    start = pd.to_datetime(last_timestamp) + pd.tseries.frequencies.to_offset(freq)
    timestamps = pd.date_range(start=start, periods=periods, freq=freq)
    return pd.DataFrame({"timestamp": timestamps})
