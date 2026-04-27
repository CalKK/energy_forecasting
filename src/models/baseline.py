from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SeasonalNaiveForecaster:
    seasonal_period: int = 24
    history_: pd.Series | None = field(default=None, init=False)

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("y must be indexed by a DatetimeIndex.")
        self.history_ = y.astype(float).copy()
        return self

    def predict_index(self, index: pd.DatetimeIndex) -> pd.Series:
        if self.history_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        preds = []
        for ts in index:
            reference = ts - pd.Timedelta(hours=self.seasonal_period)
            if reference in self.history_.index:
                preds.append(float(self.history_.loc[reference]))
            else:
                preds.append(float(self.history_.iloc[-self.seasonal_period:].mean()))
        return pd.Series(preds, index=index, name="seasonal_naive_forecast")

    def forecast(self, steps: int, freq: str = "h") -> pd.Series:
        if self.history_ is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        start = self.history_.index.max() + pd.tseries.frequencies.to_offset(freq)
        future_index = pd.date_range(start=start, periods=steps, freq=freq)
        values = list(self.history_.iloc[-self.seasonal_period:].astype(float).values)
        if len(values) == 0:
            values = [0.0]
        repeated = np.resize(values, steps)
        return pd.Series(repeated, index=future_index, name="seasonal_naive_forecast")
