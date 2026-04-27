from __future__ import annotations

from energy_modeling.config import load_settings
from energy_modeling.pipeline import run_profile, run_training


if __name__ == "__main__":
    settings = load_settings("config/default.yaml")
    run_profile(settings)
    outputs = run_training(settings)
    for key, value in outputs.items():
        print(f"{key}: {value}")
