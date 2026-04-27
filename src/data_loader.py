from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import numpy as np
import pandas as pd


def read_energy_excel(
    path: str | Path,
    sheet_name: str = "Foglio1",
    required_columns: Iterable[str] | None = None,
    sample_serials: int | None = None,
    id_col: str = "serial",
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        df = df[list(required_columns)]

    if sample_serials is not None and sample_serials > 0 and id_col in df.columns:
        serials = sorted(df[id_col].dropna().unique())[:sample_serials]
        df = df[df[id_col].isin(serials)].copy()

    return df


def clean_energy_frame(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
) -> pd.DataFrame:
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column not found: {datetime_col}")
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    cleaned = df.copy()
    cleaned[datetime_col] = pd.to_datetime(cleaned[datetime_col], errors="coerce")
    cleaned = cleaned.dropna(subset=[datetime_col])
    cleaned = cleaned.sort_values(datetime_col)

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned[target_col] = cleaned[target_col].clip(lower=0)
    return cleaned


def profile_energy_data(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    id_col: str | None = None,
    output_path: str | Path | None = None,
) -> dict:
    profile = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "datetime_min": str(df[datetime_col].min()) if datetime_col in df else None,
        "datetime_max": str(df[datetime_col].max()) if datetime_col in df else None,
        "target": target_col,
        "target_non_null": int(df[target_col].notna().sum()) if target_col in df else 0,
        "target_total": float(df[target_col].sum()) if target_col in df else None,
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "numeric_summary": {},
    }
    if id_col and id_col in df.columns:
        profile["unique_ids"] = int(df[id_col].nunique())
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        profile["numeric_summary"][col] = {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "max": float(series.max()),
        }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_path).open("w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
    return profile
