"""
Phase 0 — Smoke Tests

Tests:
1. Health endpoint returns 200 with correct JSON
2. Config loads without error and has expected defaults
3. All backend modules import cleanly
4. Root endpoint works
"""

from __future__ import annotations

import importlib

import pytest


# ──────────────────────────────────────────────
# Health endpoint
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(client):
    """GET /health should return 200 with status=ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "MulTiCheat"
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """GET / should return 200 with welcome message."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "MulTiCheat" in data["message"]


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

def test_config_loads_with_defaults():
    """Settings should instantiate with valid defaults."""
    from backend.config import Settings

    settings = Settings()
    assert settings.app_name == "MulTiCheat"
    assert settings.port == 8000
    assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    assert settings.max_upload_size_mb > 0


def test_config_upload_path_is_absolute():
    """upload_path property should return an absolute path."""
    from backend.config import Settings

    settings = Settings()
    assert settings.upload_path.is_absolute()


def test_config_evidence_path_is_absolute():
    """evidence_path property should return an absolute path."""
    from backend.config import Settings

    settings = Settings()
    assert settings.evidence_path.is_absolute()


def test_config_rejects_invalid_log_level():
    """Settings should reject an invalid log_level."""
    from backend.config import Settings

    with pytest.raises(Exception):
        Settings(log_level="INVALID_LEVEL")


def test_config_rejects_zero_upload_size():
    """Settings should reject max_upload_size_mb <= 0."""
    from backend.config import Settings

    with pytest.raises(Exception):
        Settings(max_upload_size_mb=0)


# ──────────────────────────────────────────────
# Import checks
# ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "module_name",
    [
        "backend",
        "backend.config",
        "backend.main",
        "backend.routers",
    ],
)
def test_backend_modules_import_cleanly(module_name: str):
    """All backend modules should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
