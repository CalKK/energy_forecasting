from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import joblib
import pandas as pd

from energy_modeling.config import Settings, ensure_output_dir
from energy_modeling.data_loader import clean_energy_frame, profile_energy_data, read_energy_excel
from energy_modeling.evaluation import metrics_table, regression_metrics
from energy_modeling.features import (
    aggregate_to_hourly,
    build_supervised_frame,
    complete_hourly_index,
    time_train_test_split,
)
from energy_modeling.models import LinearFeatureForecaster, SeasonalNaiveForecaster, StructuralTimeSeriesForecaster
from energy_modeling.plotting import plot_forecast, plot_forecast_bytes


def load_and_prepare(settings: Settings, df: pd.DataFrame | None = None) -> pd.DataFrame:
    data_cfg = settings.get("data")
    agg_cfg = settings.get("aggregation")
    if df is None:
        raw = read_energy_excel(
            path=settings.data_path,
            sheet_name=data_cfg.get("sheet_name", "Foglio1"),
            required_columns=data_cfg.get("required_columns"),
            sample_serials=data_cfg.get("sample_serials"),
            id_col=data_cfg.get("id_col", "serial"),
        )
    else:
        raw = df.copy()
        required_columns = data_cfg.get("required_columns")
        if required_columns:
            missing = [col for col in required_columns if col not in raw.columns]
            if missing:
                raise ValueError(f"Missing expected columns: {missing}")
            raw = raw[list(required_columns)]

        sample_serials = data_cfg.get("sample_serials")
        id_col = data_cfg.get("id_col", "serial")
        if sample_serials is not None and sample_serials > 0 and id_col in raw.columns:
            serials = sorted(raw[id_col].dropna().unique())[:sample_serials]
            raw = raw[raw[id_col].isin(serials)].copy()

    clean = clean_energy_frame(
        raw,
        datetime_col=data_cfg.get("datetime_col", "localtime_hour"),
        target_col=data_cfg.get("target_col", "kilowatt_hours"),
    )
    hourly = aggregate_to_hourly(
        clean,
        datetime_col=data_cfg.get("datetime_col", "localtime_hour"),
        target_col=data_cfg.get("target_col", "kilowatt_hours"),
        frequency=agg_cfg.get("frequency", "h"),
        target_agg=agg_cfg.get("target_agg", "sum"),
        numeric_agg=agg_cfg.get("numeric_agg", "mean"),
    )
    complete = complete_hourly_index(
        hourly,
        target_col=data_cfg.get("target_col", "kilowatt_hours"),
        target_fill=agg_cfg.get("fill_target_method", "zero"),
        exog_fill=agg_cfg.get("fill_exog_method", "interpolate"),
    )
    return complete


def run_profile(settings: Settings, df: pd.DataFrame | None = None) -> dict:
    out_dir = ensure_output_dir(settings)
    frame = load_and_prepare(settings, df=df)
    target_col = settings.get("data", "target_col", default="kilowatt_hours")
    profile_path = out_dir / settings.get("outputs", "profile_file", default="data_profile.json")
    return profile_energy_data(
        frame,
        datetime_col="timestamp",
        target_col=target_col,
        output_path=profile_path,
    )


def run_training(
    settings: Settings,
    df: pd.DataFrame | None = None,
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    out_dir = ensure_output_dir(settings) if persist_artifacts else None
    target_col = settings.get("data", "target_col", default="kilowatt_hours")
    frame = load_and_prepare(settings, df=df)

    test_days = int(settings.get("forecast", "test_days", default=14))
    train, test = time_train_test_split(frame, timestamp_col="timestamp", test_days=test_days)
    train_y = train.set_index("timestamp")[target_col].asfreq("h")
    test_y = test.set_index("timestamp")[target_col].asfreq("h")

    alpha = float(settings.get("forecast", "confidence_alpha", default=0.05))
    metric_rows = []

    baseline_period = int(settings.get("models", "baseline", "seasonal_period", default=24))
    baseline = SeasonalNaiveForecaster(seasonal_period=baseline_period).fit(train_y)
    baseline_pred = baseline.predict_index(test_y.index)
    metric_rows.append({"model": "seasonal_naive", **regression_metrics(test_y, baseline_pred)})

    feature_cfg = settings.get("features")
    exog_cols = feature_cfg.get("exogenous_columns", [])
    supervised, feature_cols = build_supervised_frame(
        frame,
        target_col=target_col,
        exogenous_cols=exog_cols,
        lags=feature_cfg.get("lags", [1, 24, 168]),
        rolling_windows=feature_cfg.get("rolling_windows", [24, 168]),
    )
    lin_train, lin_test = time_train_test_split(supervised, timestamp_col="timestamp", test_days=test_days)
    linear = LinearFeatureForecaster(
        feature_cols=feature_cols,
        target_col=target_col,
        alpha=float(settings.get("models", "linear", "alpha", default=1.0)),
    ).fit(lin_train)
    linear_pred = linear.predict(lin_test)
    linear_actual = lin_test.set_index("timestamp")[target_col]
    metric_rows.append({"model": "linear_ridge", **regression_metrics(linear_actual, linear_pred)})

    sts_cfg = settings.get("models", "structural_time_series")
    use_exog = bool(sts_cfg.get("use_exogenous", False))
    sts_exog_cols = [col for col in exog_cols if col in frame.columns] if use_exog else []
    train_exog = train.set_index("timestamp")[sts_exog_cols] if sts_exog_cols else None
    test_exog = test.set_index("timestamp")[sts_exog_cols] if sts_exog_cols else None
    sts = StructuralTimeSeriesForecaster(
        level=sts_cfg.get("level", "local level"),
        daily_period=int(sts_cfg.get("daily_period", 24)),
        weekly_period=int(sts_cfg.get("weekly_period", 168)),
        weekly_harmonics=int(sts_cfg.get("weekly_harmonics", 3)),
        maxiter=int(sts_cfg.get("maxiter", 250)),
        use_exogenous=use_exog,
        exogenous_cols=sts_exog_cols,
    ).fit(train_y, exog=train_exog)
    sts_pred = sts.predict_index(test_y.index, exog=test_exog, alpha=alpha)
    metric_rows.append({"model": "structural_time_series", **regression_metrics(test_y, sts_pred["forecast"])})

    metrics = metrics_table(metric_rows).sort_values("mae")

    forecast_eval = pd.DataFrame(
        {
            "timestamp": test_y.index,
            "actual": test_y.values,
            "seasonal_naive": baseline_pred.reindex(test_y.index).values,
            "structural_time_series": sts_pred["forecast"].reindex(test_y.index).values,
            "sts_lower": sts_pred["lower"].reindex(test_y.index).values,
            "sts_upper": sts_pred["upper"].reindex(test_y.index).values,
        }
    )
    model_bundle = {
        "selected_model": "structural_time_series",
        "model": sts,
        "baseline": baseline,
        "linear": linear,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "metrics": metrics,
    }
    if persist_artifacts:
        metrics_path = out_dir / settings.get("outputs", "metrics_file", default="metrics.csv")
        metrics.to_csv(metrics_path, index=False)

        forecast_path = out_dir / settings.get("outputs", "forecast_file", default="forecast.csv")
        forecast_eval.to_csv(forecast_path, index=False)

        plot_path = out_dir / settings.get("outputs", "plot_file", default="forecast_plot.png")
        plot_forecast(
            actual=test_y,
            forecast=sts_pred,
            output_path=plot_path,
            title="Kapelbok Structural Time Series Forecast",
        )

        model_path = out_dir / settings.get("outputs", "model_file", default="trained_model.joblib")
        joblib.dump(model_bundle, model_path)

        return {
            "metrics": metrics_path,
            "forecast": forecast_path,
            "plot": plot_path,
            "model": model_path,
        }

    model_buffer = BytesIO()
    joblib.dump(model_bundle, model_buffer)

    return {
        "metrics_df": metrics,
        "forecast_df": forecast_eval,
        "plot_bytes": plot_forecast_bytes(
            actual=test_y,
            forecast=sts_pred,
            title="Kapelbok Structural Time Series Forecast",
        ),
        "model_bytes": model_buffer.getvalue(),
    }


def run_future_forecast(settings: Settings, model_path: str | Path | None = None, horizon_hours: int | None = None) -> pd.DataFrame:
    out_dir = ensure_output_dir(settings)
    path = Path(model_path) if model_path else out_dir / settings.get("outputs", "model_file", default="trained_model.joblib")
    bundle = joblib.load(path)
    model = bundle["model"]
    horizon = int(horizon_hours or settings.get("forecast", "horizon_hours", default=168))
    forecast = model.forecast(steps=horizon, alpha=float(settings.get("forecast", "confidence_alpha", default=0.05)))
    forecast_path = out_dir / "future_forecast.csv"
    forecast.reset_index(names="timestamp").to_csv(forecast_path, index=False)
    return forecast
