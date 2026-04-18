"""
Shared pytest fixtures for MulTiCheat tests.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import create_app


@pytest.fixture()
def app():
    """Create a fresh FastAPI app instance for each test."""
    return create_app()


@pytest.fixture()
async def client(app):
    """Async HTTP client bound to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
