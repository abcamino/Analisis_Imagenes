"""Tests for analysis session endpoints."""

import pytest


class TestSessionCRUD:
    """Tests for session CRUD operations."""

    def test_create_session(self, authenticated_client):
        """Test creating new analysis session."""
        response = authenticated_client.post("/api/sessions", json={
            "name": "Test Session",
            "description": "A test analysis session"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Session"
        assert data["is_active"] is True

    def test_create_session_requires_auth(self, client):
        """Test session creation requires authentication."""
        response = client.post("/api/sessions", json={
            "name": "Test Session"
        })
        assert response.status_code == 401

    def test_list_sessions(self, authenticated_client, test_session):
        """Test listing user's sessions."""
        response = authenticated_client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(s["id"] == test_session.id for s in data)

    def test_get_session_detail(self, authenticated_client, test_session):
        """Test getting session details."""
        response = authenticated_client.get(f"/api/sessions/{test_session.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == test_session.name

    def test_end_session(self, authenticated_client, test_session):
        """Test ending an active session."""
        response = authenticated_client.post(f"/api/sessions/{test_session.id}/end")
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is False
        assert data["ended_at"] is not None

    def test_update_session(self, authenticated_client, test_session):
        """Test updating session details."""
        response = authenticated_client.put(f"/api/sessions/{test_session.id}", json={
            "name": "Updated Name"
        })
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    def test_delete_session(self, authenticated_client, test_session):
        """Test deleting a session."""
        response = authenticated_client.delete(f"/api/sessions/{test_session.id}")
        assert response.status_code == 204

        # Verify deleted
        response = authenticated_client.get(f"/api/sessions/{test_session.id}")
        assert response.status_code == 404
