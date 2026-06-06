from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    """Thin wrapper around the YAML project configuration.

    The wrapper keeps path handling centralized. All relative paths are resolved
    from the project root, never from the caller's current working directory.
    """

    raw: dict[str, Any]
    project_root: Path

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ProjectConfig":
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute() and not config_path.exists():
            project_relative = DEFAULT_CONFIG_PATH.parents[1] / config_path
            if project_relative.exists():
                config_path = project_relative
        config_path = config_path.resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        project_root = config_path.parent.parent
        return cls(raw=raw, project_root=project_root)

    def get(self, *keys: str, default: Any = None) -> Any:
        obj: Any = self.raw
        for key in keys:
            if not isinstance(obj, Mapping) or key not in obj:
                return default
            obj = obj[key]
        return obj

    def path(self, *keys: str) -> Path:
        value = self.get(*keys)
        if value is None:
            raise KeyError(f"Missing path config: {'.'.join(keys)}")
        p = Path(value)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def ensure_dirs(self) -> None:
        for key in [
            "output_dir",
            "feature_dir",
            "prediction_dir",
            "evaluation_dir",
            "check_dir",
            "model_dir",
            "report_html_dir",
            "log_dir",
        ]:
            self.path("paths", key).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig.load(config_path)
    cfg.ensure_dirs()
    return cfg
