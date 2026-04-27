from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearFeatureForecaster:
    feature_cols: list[str]
    target_col: str
    alpha: float = 1.0
    pipeline_: Pipeline | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame) -> "LinearFeatureForecaster":
        missing = [col for col in self.feature_cols + [self.target_col] if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns for linear model: {missing}")
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[("numeric", numeric_transformer, self.feature_cols)],
            remainder="drop",
        )
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline_.fit(frame[self.feature_cols], frame[self.target_col])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        preds = self.pipeline_.predict(frame[self.feature_cols])
        index = pd.to_datetime(frame["timestamp"]) if "timestamp" in frame.columns else frame.index
        return pd.Series(preds, index=index, name="linear_forecast").clip(lower=0)
