from __future__ import annotations

import argparse
from pathlib import Path
from energy_modeling.config import load_settings
from energy_modeling.pipeline import run_future_forecast, run_profile, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kapelbok hourly energy forecasting workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Create a JSON profile of the cleaned hourly dataset")
    profile.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration file")

    train = subparsers.add_parser("train", help="Train and evaluate baseline, linear, and STS models")
    train.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration file")

    forecast = subparsers.add_parser("forecast", help="Generate future forecasts from a saved STS model")
    forecast.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration file")
    forecast.add_argument("--model", default=None, help="Optional path to saved joblib model bundle")
    forecast.add_argument("--horizon", type=int, default=None, help="Forecast horizon in hours")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(args.config)

    if args.command == "profile":
        profile = run_profile(settings)
        print(f"Profile created for {profile['rows']:,} hourly records.")
        print(f"Date range: {profile['datetime_min']} to {profile['datetime_max']}")
        print(f"Output directory: {settings.output_dir}")
    elif args.command == "train":
        outputs = run_training(settings)
        print("Training complete.")
        for name, path in outputs.items():
            print(f"{name}: {Path(path)}")
    elif args.command == "forecast":
        forecast = run_future_forecast(settings, model_path=args.model, horizon_hours=args.horizon)
        print(f"Generated {len(forecast):,} forecast rows.")
        print(forecast.head().to_string())


if __name__ == "__main__":
    main()
