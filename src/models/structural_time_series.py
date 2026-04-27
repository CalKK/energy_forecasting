from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


@dataclass
class StructuralTimeSeriesForecaster:
    level: str = "local level"
    daily_period: int = 24
    weekly_period: int = 168
    weekly_harmonics: int = 3
    maxiter: int = 250
    use_exogenous: bool = False
    exogenous_cols: list[str] | None = None
    model_: Any = field(default=None, init=False)
    results_: Any = field(default=None, init=False)
    train_index_: pd.DatetimeIndex | None = field(default=None, init=False)

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "StructuralTimeSeriesForecaster":
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("y must be indexed by a DatetimeIndex.")
        y = y.astype(float).asfreq("h")
        y = y.interpolate(limit_direction="both").fillna(0)
        self.train_index_ = y.index

        exog_fit = None
        if self.use_exogenous and self.exogenous_cols:
            if exog is None:
                raise ValueError("use_exogenous=True requires an exogenous dataframe.")
            exog_fit = exog.reindex(y.index)[self.exogenous_cols].interpolate(limit_direction="both").ffill().bfill()

        freq_components = []
        if self.weekly_period and self.weekly_harmonics:
            freq_components.append({"period": self.weekly_period, "harmonics": self.weekly_harmonics})

        self.model_ = UnobservedComponents(
            endog=y,
            level=self.level,
            seasonal=self.daily_period,
            freq_seasonal=freq_components or None,
            exog=exog_fit,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results_ = self.model_.fit(method="lbfgs", maxiter=self.maxiter, disp=False)
        return self

    def predict_index(
        self,
        index: pd.DatetimeIndex,
        exog: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        if self.results_ is None or self.train_index_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        start = index.min()
        end = index.max()
        exog_pred = None
        if self.use_exogenous and self.exogenous_cols:
            if exog is None:
                raise ValueError("Prediction requires future/test exogenous dataframe.")
            exog_pred = exog.reindex(index)[self.exogenous_cols].interpolate(limit_direction="both").ffill().bfill()

        pred = self.results_.get_prediction(start=start, end=end, exog=exog_pred)
        frame = pd.DataFrame({"forecast": pred.predicted_mean}, index=pred.predicted_mean.index)
        conf = pred.conf_int(alpha=alpha)
        frame["lower"] = conf.iloc[:, 0]
        frame["upper"] = conf.iloc[:, 1]
        return frame.reindex(index).clip(lower=0)

    def forecast(
        self,
        steps: int,
        exog_future: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        if self.results_ is None or self.train_index_ is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        exog_fc = None
        if self.use_exogenous and self.exogenous_cols:
            if exog_future is None:
                raise ValueError("Forecasting with exogenous variables requires exog_future.")
            exog_fc = exog_future[self.exogenous_cols]
        pred = self.results_.get_forecast(steps=steps, exog=exog_fc)
        out = pd.DataFrame({"forecast": pred.predicted_mean}, index=pred.predicted_mean.index)
        conf = pred.conf_int(alpha=alpha)
        out["lower"] = conf.iloc[:, 0]
        out["upper"] = conf.iloc[:, 1]
        return out.clip(lower=0)
