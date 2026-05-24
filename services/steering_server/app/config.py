from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatabaseSettings:
    url: str = f"sqlite:///{ROOT / 'steering_server.db'}"


@dataclass(frozen=True)
class AuthSettings:
    bootstrap_user: str = "local-admin"
    bootstrap_api_key: str = "dev-steering-key"


@dataclass(frozen=True)
class RateLimitSettings:
    default_limit: str = "120/minute"
    steer_limit: str = "30/minute"


@dataclass(frozen=True)
class TelemetrySettings:
    service_name: str = "steering-server"
    otlp_endpoint: str = "http://otel-collector:4317"


@dataclass(frozen=True)
class QueueSettings:
    run_inline: bool = True


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings = DatabaseSettings()
    auth: AuthSettings = AuthSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    telemetry: TelemetrySettings = TelemetrySettings()
    queue: QueueSettings = QueueSettings()


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings() -> Settings:
    data = _merge(_read_toml(ROOT / "config" / "default.toml"), _read_toml(ROOT / "config" / "local.toml"))
    database_data = data.get("database", {})
    database_url = database_data.get("url")
    if isinstance(database_url, str) and database_url.startswith("sqlite:///") and not database_url.startswith("sqlite:////"):
        raw_path = database_url[len("sqlite:///") :]
        if raw_path and not Path(raw_path).is_absolute():
            database_data = dict(database_data)
            database_data["url"] = f"sqlite:///{ROOT / raw_path}"
    return Settings(
        database=DatabaseSettings(**database_data),
        auth=AuthSettings(**data.get("auth", {})),
        rate_limit=RateLimitSettings(**data.get("rate_limit", {})),
        telemetry=TelemetrySettings(**data.get("telemetry", {})),
        queue=QueueSettings(**data.get("queue", {})),
    )


settings = load_settings()
