"""Tests for camera API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_cameras_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Test listing cameras when none exist."""
    response = await client.get("/api/cameras", headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data["cameras"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_create_camera(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test creating a new camera (admin only)."""
    camera_data = {
        "name": "Front Door",
        "host": "192.168.1.100",
        "port": 554,
        "path": "/cam/realmonitor",
        "username": "admin",
        "password": "secret",
        "transport": "tcp",
        "recording_enabled": True,
    }

    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    assert response.status_code == 201

    data = response.json()
    assert data["id"] is not None
    assert data["name"] == "Front Door"
    assert data["host"] == "192.168.1.100"
    assert data["port"] == 554
    assert data["status"] == "unknown"
    # Password should not be in response
    assert "password" not in data or data.get("password") is None


@pytest.mark.asyncio
async def test_create_camera_minimal(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test creating a camera with minimal data (admin only)."""
    camera_data = {
        "name": "Backyard",
        "host": "192.168.1.101",
    }

    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Backyard"
    assert data["port"] == 554  # Default
    assert data["path"] == "/cam/realmonitor"  # Default
    assert data["transport"] == "tcp"  # Default


@pytest.mark.asyncio
async def test_create_camera_duplicate_name(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test creating a camera with duplicate name fails."""
    camera_data = {"name": "Duplicate Test", "host": "192.168.1.100"}

    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    assert response.status_code == 201

    # Try to create another with same name
    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_camera_invalid_transport(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test creating a camera with invalid transport fails."""
    camera_data = {
        "name": "Invalid Transport",
        "host": "192.168.1.100",
        "transport": "invalid",
    }

    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_create_camera_path_normalization(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test that path is normalized to start with /."""
    camera_data = {
        "name": "Path Test",
        "host": "192.168.1.100",
        "path": "stream/main",  # Missing leading /
    }

    response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    assert response.status_code == 201
    assert response.json()["path"] == "/stream/main"


@pytest.mark.asyncio
async def test_get_camera(
    client: AsyncClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
) -> None:
    """Test getting a specific camera."""
    # First create a camera (requires admin)
    camera_data = {"name": "Get Test", "host": "192.168.1.100"}
    create_response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    camera_id = create_response.json()["id"]

    # Then get it (regular user can read)
    response = await client.get(f"/api/cameras/{camera_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["name"] == "Get Test"


@pytest.mark.asyncio
async def test_get_camera_not_found(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Test getting a non-existent camera."""
    response = await client.get("/api/cameras/99999", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_camera(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test updating a camera (admin only)."""
    # Create a camera
    camera_data = {"name": "Update Test", "host": "192.168.1.100"}
    create_response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    camera_id = create_response.json()["id"]

    # Update it
    update_data = {"name": "Updated Name", "port": 8554}
    response = await client.put(
        f"/api/cameras/{camera_id}", json=update_data, headers=admin_headers
    )
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Updated Name"
    assert data["port"] == 8554
    assert data["host"] == "192.168.1.100"  # Unchanged


@pytest.mark.asyncio
async def test_update_camera_partial(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test partial update of a camera (admin only)."""
    # Create a camera
    camera_data = {
        "name": "Partial Update",
        "host": "192.168.1.100",
        "port": 554,
    }
    create_response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    camera_id = create_response.json()["id"]

    # Update only recording_enabled
    update_data = {"recording_enabled": False}
    response = await client.put(
        f"/api/cameras/{camera_id}", json=update_data, headers=admin_headers
    )
    assert response.status_code == 200

    data = response.json()
    assert data["recording_enabled"] is False
    assert data["name"] == "Partial Update"  # Unchanged
    assert data["port"] == 554  # Unchanged


@pytest.mark.asyncio
async def test_update_camera_not_found(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test updating a non-existent camera."""
    response = await client.put(
        "/api/cameras/99999", json={"name": "Test"}, headers=admin_headers
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_camera_duplicate_name(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test updating a camera to a duplicate name fails."""
    # Create two cameras
    await client.post(
        "/api/cameras",
        json={"name": "Camera A", "host": "192.168.1.1"},
        headers=admin_headers,
    )
    response = await client.post(
        "/api/cameras",
        json={"name": "Camera B", "host": "192.168.1.2"},
        headers=admin_headers,
    )
    camera_b_id = response.json()["id"]

    # Try to rename Camera B to Camera A
    response = await client.put(
        f"/api/cameras/{camera_b_id}",
        json={"name": "Camera A"},
        headers=admin_headers,
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_delete_camera(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test deleting a camera (admin only)."""
    # Create a camera
    camera_data = {"name": "Delete Test", "host": "192.168.1.100"}
    create_response = await client.post(
        "/api/cameras", json=camera_data, headers=admin_headers
    )
    camera_id = create_response.json()["id"]

    # Delete it
    response = await client.delete(
        f"/api/cameras/{camera_id}", headers=admin_headers
    )
    assert response.status_code == 204

    # Verify it's gone
    response = await client.get(f"/api/cameras/{camera_id}", headers=admin_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_camera_not_found(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    """Test deleting a non-existent camera."""
    response = await client.delete("/api/cameras/99999", headers=admin_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_cameras_with_data(
    client: AsyncClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
) -> None:
    """Test listing cameras returns all cameras."""
    # Create multiple cameras (requires admin)
    for i in range(3):
        await client.post(
            "/api/cameras",
            json={"name": f"Camera {i}", "host": f"192.168.1.{i}"},
            headers=admin_headers,
        )

    # List cameras (regular user can read)
    response = await client.get("/api/cameras", headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 3
    assert len(data["cameras"]) == 3


@pytest.mark.asyncio
async def test_test_camera_not_found(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Test testing a non-existent camera."""
    response = await client.post("/api/cameras/99999/test", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unauthorized_access(client: AsyncClient) -> None:
    """Test that endpoints require authentication."""
    response = await client.get("/api/cameras")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_camera_requires_admin(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Test that creating a camera requires admin privileges."""
    camera_data = {"name": "Non-Admin Camera", "host": "192.168.1.100"}
    response = await client.post(
        "/api/cameras", json=camera_data, headers=auth_headers
    )
    assert response.status_code == 403
