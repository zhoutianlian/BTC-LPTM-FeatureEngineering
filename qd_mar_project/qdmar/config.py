"""Configuration helpers for QD-MAR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass(frozen=True)
class HorizonConfig:
    name: str
    minutes: int
    ret_col: str
    plie_col: str
    raw_mag_col: str
    eff_mag_col: str
    b_min_bps: float


class Config:
    """Thin wrapper around the YAML configuration.

    The wrapper keeps path handling centralized and avoids hard-coded output
    locations inside the algorithm modules.
    """

    def __init__(self, data: Dict[str, Any], root_dir: Path):
        self.data = data
        self.root_dir = root_dir

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        project_cfg = data.get("project", {}) if isinstance(data, dict) else {}
        configured_root = project_cfg.get("root_dir")
        if configured_root:
            root_dir = Path(configured_root)
            if not root_dir.is_absolute():
                root_dir = (path.parent / root_dir).resolve()
        else:
            root_dir = path.parent.parent
        return cls(data=data, root_dir=root_dir)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, *keys: str) -> Path:
        rel = self.get(*keys)
        if rel is None:
            raise KeyError("Missing config path: " + ".".join(keys))
        return self.resolve_path(rel)

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a configured path relative to the project root."""
        p = Path(value)
        return p if p.is_absolute() else (self.root_dir / p)

    def output_path(self, section: str, name: str, default: str | Path) -> Path:
        """Resolve an output artifact path from ``outputs.<section>.<name>``.

        The config may store either a project-relative path or a filename.  A
        bare filename is resolved inside the corresponding configured output
        directory (``paths.features_dir``, ``paths.reports_dir`` or
        ``paths.html_dir``).
        """
        configured = self.get("outputs", section, name)
        value = Path(configured if configured is not None else default)
        if value.is_absolute():
            return value
        if len(value.parts) > 1:
            return self.root_dir / value
        base_key = {
            "features": "features_dir",
            "reports": "reports_dir",
            "html": "html_dir",
        }.get(section)
        if base_key:
            return self.path("paths", base_key) / value
        return self.root_dir / value

    def output_columns(self, name: str, default: list[str]) -> list[str]:
        return list(self.get("outputs", "columns", name, default=default) or [])

    def output_prefixes(self, name: str, default: list[str]) -> tuple[str, ...]:
        values = self.get("outputs", "column_prefixes", name, default=default) or []
        return tuple(str(v) for v in values)

    def run_flag(self, name: str, default: Any = None) -> Any:
        return self.get("run", name, default=default)

    def io_option(self, name: str, default: Any = None) -> Any:
        return self.get("io", name, default=default)

    def html_filename(self, name: str, default: str) -> str:
        return str(self.get("outputs", "html", name, default=default))

    @property
    def horizons(self) -> List[HorizonConfig]:
        return [HorizonConfig(**h) for h in self.get("horizons")]

    @property
    def agent_inputs(self) -> list[str]:
        return list(self.get("agent_inputs", default=[]))

    def ensure_dirs(self) -> None:
        for k in ["output_dir", "features_dir", "reports_dir", "html_dir", "state_dir"]:
            self.path("paths", k).mkdir(parents=True, exist_ok=True)
