"""
MulTiCheat — Application Configuration

Loads settings from environment variables (prefixed MULTICHEAT_) and config.yaml.
Uses pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Resolve project root (parent of the backend/ package)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml from the project root. Returns empty dict on failure."""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        logger.warning("config.yaml not found at %s — using defaults.", config_path)
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config.yaml from %s", config_path)
        return data
    except Exception as exc:
        logger.error("Failed to parse config.yaml: %s", exc)
        return {}


# Pre-load YAML so we can inject values as defaults
_yaml = _load_yaml_config()


class Settings(BaseSettings):
    """Central application settings.

    Priority (highest → lowest):
      1. Environment variables  (MULTICHEAT_*)
      2. .env file
      3. config.yaml defaults
      4. Field defaults below
    """

    model_config = SettingsConfigDict(
        env_prefix="MULTICHEAT_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ────────────────────────────────────
    app_name: str = Field(
        default=_yaml.get("app", {}).get("name", "MulTiCheat"),
        description="Application display name",
    )
    app_version: str = Field(
        default=_yaml.get("app", {}).get("version", "0.1.0"),
        description="Application version string",
    )
    debug: bool = Field(
        default=_yaml.get("app", {}).get("debug", False),
        description="Enable debug mode",
    )

    # ── Server ─────────────────────────────────
    host: str = Field(
        default=_yaml.get("server", {}).get("host", "0.0.0.0"),
        description="Server bind host",
    )
    port: int = Field(
        default=_yaml.get("server", {}).get("port", 8000),
        description="Server bind port",
    )

    # ── Storage ────────────────────────────────
    upload_dir: str = Field(
        default=_yaml.get("storage", {}).get("upload_dir", "uploads"),
        description="Directory for uploaded videos",
    )
    evidence_dir: str = Field(
        default=_yaml.get("storage", {}).get("evidence_dir", "evidence"),
        description="Directory for evidence images",
    )
    max_upload_size_mb: int = Field(
        default=_yaml.get("storage", {}).get("max_upload_size_mb", 500),
        description="Maximum upload file size in MB",
    )

    # ── Logging ────────────────────────────────
    log_level: str = Field(
        default=_yaml.get("logging", {}).get("level", "INFO"),
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_format: str = Field(
        default=_yaml.get("logging", {}).get(
            "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ),
        description="Logging format string",
    )

    # ── Validators ─────────────────────────────
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper

    @field_validator("max_upload_size_mb")
    @classmethod
    def _validate_max_upload_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_upload_size_mb must be > 0")
        return v

    # ── Helpers ────────────────────────────────
    @property
    def upload_path(self) -> Path:
        """Resolved absolute path for uploads directory."""
        p = Path(self.upload_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def evidence_path(self) -> Path:
        """Resolved absolute path for evidence directory."""
        p = Path(self.evidence_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def ensure_directories(self) -> None:
        """Create upload and evidence directories if they don't exist."""
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.evidence_path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directories: %s, %s", self.upload_path, self.evidence_path)


def get_settings() -> Settings:
    """Factory that returns a cached Settings singleton."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()  # type: ignore[attr-defined]
    return get_settings._instance  # type: ignore[attr-defined]
