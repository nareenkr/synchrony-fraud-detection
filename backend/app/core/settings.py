"""Validated environment-backed settings without a framework-specific dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Boolean settings must be true/false, yes/no, on/off, or 1/0")


@dataclass(frozen=True, slots=True)
class Settings:
    app_env: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    database_url: str = "sqlite:///./synchrony.db"
    persistence_namespace: str = "demo"
    state_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    pseudonym_key: str = "local-development-only-change-me"
    model_bundle_path: Path = Path("artifacts/supervised-v1")
    anomaly_artifact_path: Path = Path("artifacts/anomaly-v1.joblib")
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    risk_config_path: Path = Path("config/risk.yaml")
    auth_enabled: bool = False
    api_read_key: str = ""
    api_admin_key: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            app_env=os.getenv("APP_ENV", "local").strip().lower(),
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./synchrony.db"),
            persistence_namespace=os.getenv("PERSISTENCE_NAMESPACE", "demo").strip().lower(),
            state_backend=os.getenv("STATE_BACKEND", "memory").strip().lower(),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            pseudonym_key=os.getenv("PSEUDONYM_KEY", "local-development-only-change-me"),
            model_bundle_path=Path(os.getenv("MODEL_BUNDLE_PATH", "artifacts/supervised-v1")),
            anomaly_artifact_path=Path(
                os.getenv("ANOMALY_ARTIFACT_PATH", "artifacts/anomaly-v1.joblib")
            ),
            cors_origins=_csv(os.getenv("CORS_ORIGINS", "http://localhost:5173")),
            risk_config_path=Path(os.getenv("RISK_CONFIG_PATH", "config/risk.yaml")),
            auth_enabled=_bool(os.getenv("AUTH_ENABLED", "false")),
            api_read_key=os.getenv("API_READ_KEY", ""),
            api_admin_key=os.getenv("API_ADMIN_KEY", ""),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.app_env not in {"local", "test", "production"}:
            raise ValueError("APP_ENV must be local, test, or production")
        if not 1 <= self.app_port <= 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")
        if self.state_backend not in {"memory", "redis"}:
            raise ValueError("STATE_BACKEND must be memory or redis")
        if not self.persistence_namespace or len(self.persistence_namespace) > 64:
            raise ValueError("PERSISTENCE_NAMESPACE must contain 1 to 64 characters")
        if not self.cors_origins:
            raise ValueError("At least one CORS origin is required")
        if self.app_env != "local" and len(self.pseudonym_key) < 32:
            raise ValueError("PSEUDONYM_KEY must contain at least 32 characters outside local mode")
        if self.app_env != "local" and self.pseudonym_key == "local-development-only-change-me":
            raise ValueError("PSEUDONYM_KEY cannot use the local default outside local mode")
        if self.app_env == "production" and not self.auth_enabled:
            raise ValueError("AUTH_ENABLED must be true in production")
        if self.auth_enabled:
            if len(self.api_read_key) < 32 or len(self.api_admin_key) < 32:
                raise ValueError("API_READ_KEY and API_ADMIN_KEY must each contain 32+ characters")
            if self.api_read_key == self.api_admin_key:
                raise ValueError("API_READ_KEY and API_ADMIN_KEY must be distinct")
