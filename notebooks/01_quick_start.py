from pathlib import Path
import pandas as pd
from energy_modeling.config import load_settings
from energy_modeling.pipeline import load_and_prepare, run_training

settings = load_settings("../config/default.yaml")
frame = load_and_prepare(settings)
print(frame.head())
print(frame.tail())
print(frame.describe())
outputs = run_training(settings)
print(outputs)
metrics = pd.read_csv(outputs["metrics"])
print(metrics)
