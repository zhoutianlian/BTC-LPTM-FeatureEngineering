from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


NON_FEATURE_COLUMNS = {"time", "price_feature_time"}


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    category: str
    description: str = ""
    documented: bool = False
    actual_output: bool = False
    important: bool = False
    expected_numeric: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def safe_feature_filename(feature_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_name).strip("._")
    return safe or "feature"


def extract_feature_definitions_from_markdown(doc_path: str | Path) -> list[FeatureDefinition]:
    path = Path(doc_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    sections = _split_numbered_sections(text)
    definitions: list[FeatureDefinition] = []
    seen: set[str] = set()
    for title, body in sections:
        description = _section_description(body)
        for name in _extract_output_names(body):
            if name in seen:
                continue
            seen.add(name)
            definitions.append(
                FeatureDefinition(
                    name=name,
                    category=title,
                    description=description,
                    documented=True,
                    important=True,
                    expected_numeric=True,
                )
            )
    return definitions


def build_feature_catalog(
    df: pd.DataFrame,
    documented: Iterable[FeatureDefinition],
    include_actual_output_features: bool = True,
) -> list[FeatureDefinition]:
    catalog: list[FeatureDefinition] = []
    by_name: dict[str, FeatureDefinition] = {}
    actual_feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    actual = set(actual_feature_names)

    for item in documented:
        merged = FeatureDefinition(
            name=item.name,
            category=item.category,
            description=item.description,
            documented=True,
            actual_output=item.name in actual,
            important=True,
            expected_numeric=item.expected_numeric,
        )
        by_name[item.name] = merged
        catalog.append(merged)

    if include_actual_output_features:
        for name in actual_feature_names:
            if name in by_name:
                continue
            catalog.append(
                FeatureDefinition(
                    name=name,
                    category=_infer_category_from_name(name),
                    description="Actual output column not explicitly listed in the feature-engineering document.",
                    documented=False,
                    actual_output=True,
                    important=False,
                    expected_numeric=True,
                )
            )
    return catalog


def _split_numbered_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+\d+\.\s+(.+?)\s*$", text))
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((title, text[start:end]))
    return sections


def _extract_output_names(section_body: str) -> list[str]:
    output_field_marker = "\u8f93\u51fa\u5b57\u6bb5"
    output_marker = "\u8f93\u51fa\uff1a"
    names: list[str] = []
    in_output = False
    seen_any = False
    for line in section_body.splitlines():
        stripped = line.strip()
        if _is_output_heading(stripped, output_field_marker, output_marker):
            in_output = True
            seen_any = False
            continue
        if not in_output:
            continue
        if not stripped:
            if seen_any:
                in_output = False
            continue
        if stripped.startswith("-"):
            for code in re.findall(r"`([^`]+)`", line):
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", code):
                    names.append(code)
                    seen_any = True
        elif seen_any:
            in_output = False
    return names


def _is_output_heading(stripped_line: str, output_field_marker: str, output_marker: str) -> bool:
    if stripped_line.startswith("-"):
        return False
    normalized = stripped_line.replace(" ", "")
    if normalized in {
        f"{output_field_marker}\uff1a",
        f"{output_field_marker}:",
        output_marker,
        "\u8f93\u51fa:",
    }:
        return True
    return normalized.endswith(f"{output_field_marker}\uff1a") or normalized.endswith(f"{output_field_marker}:") or normalized.endswith(output_marker) or normalized.endswith("\u8f93\u51fa:")


def _section_description(section_body: str) -> str:
    lines: list[str] = []
    in_code = False
    for raw in section_body.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            if lines:
                break
            continue
        if line.startswith("#") or line.startswith("|") or line.startswith("-"):
            continue
        if "=" in line and len(line) < 120:
            continue
        lines.append(line)
        if len(" ".join(lines)) > 420:
            break
    return " ".join(lines)[:600]


def _infer_category_from_name(name: str) -> str:
    prefixes = [
        ("past_return_", "Past return"),
        ("realized_vol_", "Realized volatility"),
        ("range_width_", "Range width"),
        ("range_compression_", "Range compression"),
        ("range_to_vol_", "Range to vol"),
        ("trend_", "Trend"),
        ("bar_direction_align_", "Bar direction align"),
        ("block_direction_align_", "Block direction align"),
        ("vol_of_vol_", "Vol of vol"),
        ("jump_", "Jump"),
        ("max_jump_z_", "Robust jump z-score"),
        ("signed_max_jump_return_", "Jump signed return"),
        ("price_missing_ratio_", "Price quality"),
        ("price_gap_flag_", "Price quality"),
        ("price_outlier_flag_", "Price quality"),
        ("price_obs_count_", "Price quality"),
        ("price_expected_count_", "Price quality"),
        ("price_feature_age_", "Feature freshness"),
    ]
    for prefix, category in prefixes:
        if name.startswith(prefix):
            return category
    return "Actual output extra"
