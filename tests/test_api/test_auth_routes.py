"""Tests for authentication endpoints."""

import pytest


class TestRegistration:
    """Tests for user registration."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post("/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "SecurePass123",
            "full_name": "New User"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert "password" not in data
        assert "hashed_password" not in data

    def test_register_duplicate_username(self, client, test_user):
        """Test registration with existing username fails."""
        response = client.post("/auth/register", json={
            "username": "testuser",
            "email": "different@example.com",
            "password": "SecurePass123"
        })
        assert response.status_code == 400
        assert "username" in response.json()["detail"].lower()

    def test_register_duplicate_email(self, client, test_user):
        """Test registration with existing email fails."""
        response = client.post("/auth/register", json={
            "username": "differentuser",
            "email": "test@example.com",
            "password": "SecurePass123"
        })
        assert response.status_code == 400
        assert "email" in response.json()["detail"].lower()

    def test_register_weak_password(self, client):
        """Test registration with weak password fails."""
        response = client.post("/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "weak"
        })
        assert response.status_code == 422

    def test_register_password_no_uppercase(self, client):
        """Test password without uppercase fails validation."""
        response = client.post("/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "lowercase123"
        })
        assert response.status_code == 422


class TestLogin:
    """Tests for user login."""

    def test_login_success(self, client, test_user):
        """Test successful login sets session cookie."""
        response = client.post("/auth/login", data={
            "username": "testuser",
            "password": "TestPass123"
        }, follow_redirects=False)
        assert response.status_code == 302
        assert "session_token" in response.cookies

    def test_login_invalid_password(self, client, test_user):
        """Test login with wrong password fails."""
        response = client.post("/auth/login", data={
            "username": "testuser",
            "password": "WrongPassword123"
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user fails."""
        response = client.post("/auth/login", data={
            "username": "nonexistent",
            "password": "SomePassword123"
        })
        assert response.status_code == 401


class TestCurrentUser:
    """Tests for current user endpoint."""

    def test_me_authenticated(self, authenticated_client, test_user):
        """Test /auth/me returns current user."""
        response = authenticated_client.get("/auth/me")
        assert response.status_code == 200
        assert response.json()["username"] == test_user.username

    def test_me_unauthenticated(self, client):
        """Test /auth/me without session returns 401."""
        response = client.get("/auth/me")
        assert response.status_code == 401
