from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml

from liq_data_download.config import CredentialsConfig
from liq_data_download.logging_utils import log_message


class DataDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class GlassnodeCredentials:
    api_key: str | None
    base_url: str
    requests_per_minute: int


@dataclass(frozen=True)
class SeriesHealth:
    name: str
    rows: int
    start_time: str | None
    latest_time: str | None
    missing_bars: int
    missing_ratio_pct: float
    stale_hours: float | None


class GlassnodeClient:
    def __init__(self, *, credentials: GlassnodeCredentials, logger=None) -> None:
        self.credentials = credentials
        self.logger = logger
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        rpm = max(int(self.credentials.requests_per_minute), 1)
        interval = 60.0 / rpm
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def request_json(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        query = dict(params)
        masked = dict(query)
        if self.credentials.api_key:
            query["api_key"] = self.credentials.api_key
            masked["api_key"] = "***"
        url = f"{self.credentials.base_url.rstrip('/')}/{endpoint.lstrip('/')}?{urlencode(query)}"
        masked_url = f"{self.credentials.base_url.rstrip('/')}/{endpoint.lstrip('/')}?{urlencode(masked, safe='*')}"
        log_message(self.logger, f"Requesting Glassnode endpoint={endpoint} params={masked}")
        self._throttle()
        try:
            response = requests.get(url, headers={"Accept": "application/json"}, timeout=(15, 45))
        except requests.RequestException as exc:
            message = str(exc)
            if self.credentials.api_key:
                message = message.replace(self.credentials.api_key, "***")
            raise DataDownloadError(
                f"Glassnode request failed endpoint={endpoint} error={exc.__class__.__name__}: {message} url={masked_url}"
            ) from None
        self._last_request_time = time.time()
        if not response.ok:
            raise DataDownloadError(
                f"Glassnode request failed endpoint={endpoint} status={response.status_code} "
                f"reason={response.reason} url={masked_url}"
            )
        payload = response.json()
        if not isinstance(payload, list):
            raise DataDownloadError(f"Unexpected response type from Glassnode: {type(payload)}")
        log_message(self.logger, f"Received endpoint={endpoint} rows={len(payload):,}")
        return payload

    def get_series(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        value_name: str,
        value_fields: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        payload = self.request_json(endpoint, params)
        if not payload:
            raise DataDownloadError(f"Empty response from Glassnode endpoint={endpoint}")
        df = pd.DataFrame(payload)
        if "t" not in df.columns:
            raise DataDownloadError(f"Response missing t column for endpoint={endpoint}")

        field_map = dict(value_fields or {})
        if field_map:
            object_columns = [
                column
                for column in df.columns
                if column != "t" and df[column].dropna().map(lambda value: isinstance(value, dict)).any()
            ]
            container = "v" if "v" in object_columns else (object_columns[0] if len(object_columns) == 1 else None)
            if container:
                rows: list[dict[str, Any]] = []
                for value in df[container]:
                    if not isinstance(value, dict):
                        raise DataDownloadError(f"Expected object values for endpoint={endpoint} metric={value_name}")
                    missing = [source for source in field_map if source not in value]
                    if missing:
                        raise DataDownloadError(f"Response missing value fields {missing} for endpoint={endpoint}")
                    rows.append({target: value[source] for source, target in field_map.items()})
                values = pd.DataFrame(rows, index=df.index)
            else:
                missing = [source for source in field_map if source not in df.columns]
                if missing:
                    raise DataDownloadError(f"Response missing value fields {missing} for endpoint={endpoint}")
                values = df[list(field_map)].rename(columns=field_map)
            out = pd.concat([df[["t"]].copy(), values], axis=1)
            out = out.rename(columns={"t": "time"})
        else:
            if "v" not in df.columns:
                raise DataDownloadError(f"Response missing v column for endpoint={endpoint}")
            out = df[["t", "v"]].copy()
            out.columns = ["time", value_name]

        out["time"] = pd.to_datetime(out["time"], unit="s", utc=True, errors="coerce").dt.tz_convert(None)
        for column in out.columns:
            if column != "time":
                out[column] = pd.to_numeric(out[column], errors="coerce")
        out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        return out.reset_index(drop=True)


def load_glassnode_credentials(cfg: CredentialsConfig, *, project_root: Path) -> GlassnodeCredentials:
    config_path = project_root / cfg.config_filename
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("glassnode", {}) if isinstance(payload, dict) else {}
    api_key = os.getenv(cfg.env_var) or section.get("api_key")
    base_url = section.get("base_url", cfg.base_url)
    rpm = int(section.get("requests_per_minute", cfg.requests_per_minute))
    return GlassnodeCredentials(api_key=api_key, base_url=base_url, requests_per_minute=rpm)


def series_health(df: pd.DataFrame, *, name: str, freq: str) -> SeriesHealth:
    ordered = df.copy()
    ordered["time"] = pd.to_datetime(ordered["time"], errors="coerce")
    ordered = ordered.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    if ordered.empty:
        return SeriesHealth(name=name, rows=0, start_time=None, latest_time=None, missing_bars=0, missing_ratio_pct=0.0, stale_hours=None)
    expected = pd.date_range(start=ordered["time"].iloc[0], end=ordered["time"].iloc[-1], freq=freq)
    missing = expected.difference(pd.DatetimeIndex(ordered["time"]))
    stale_hours = float((pd.Timestamp.utcnow().tz_localize(None) - ordered["time"].iloc[-1]) / pd.Timedelta(hours=1))
    return SeriesHealth(
        name=name,
        rows=int(len(ordered)),
        start_time=str(ordered["time"].iloc[0]),
        latest_time=str(ordered["time"].iloc[-1]),
        missing_bars=int(len(missing)),
        missing_ratio_pct=float(len(missing) / max(len(expected), 1) * 100.0),
        stale_hours=stale_hours,
    )
