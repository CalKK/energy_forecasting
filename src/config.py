from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    base_dir: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    @property
    def output_dir(self) -> Path:
        output = Path(self.get("outputs", "directory", default="outputs"))
        return output if output.is_absolute() else self.base_dir / output

    @property
    def data_path(self) -> Path:
        data_path = Path(self.get("data", "path"))
        return data_path if data_path.is_absolute() else self.base_dir / data_path


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} did not parse into a dictionary.")
    return Settings(raw=raw, base_dir=path.parent.parent if path.parent.name == "config" else path.parent)


def ensure_output_dir(settings: Settings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir
