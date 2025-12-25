"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient

from app import __version__


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """Test root endpoint returns API info."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "RoninNVR"
    assert data["version"] == __version__
    assert "docs" in data
    assert "health" in data


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """Test health check endpoint."""
    response = await client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["database"] == "ok"


@pytest.mark.asyncio
async def test_liveness_endpoint(client: AsyncClient) -> None:
    """Test liveness probe endpoint."""
    response = await client.get("/api/health/live")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "alive"
