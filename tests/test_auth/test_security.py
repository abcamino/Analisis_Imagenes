"""Tests for authentication security functions."""

import pytest
from webapp.auth.security import (
    hash_password,
    verify_password,
    create_session_token,
    validate_password_strength
)


class TestPasswordHashing:
    """Tests for password hashing."""

    def test_hash_password(self):
        """Test password is hashed."""
        password = "SecurePass123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 20

    def test_verify_password_correct(self):
        """Test correct password verification."""
        password = "SecurePass123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test incorrect password verification."""
        hashed = hash_password("SecurePass123")

        assert verify_password("WrongPassword", hashed) is False

    def test_hash_is_random(self):
        """Test same password produces different hashes (salt)."""
        password = "SecurePass123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2


class TestSessionManagement:
    """Tests for session token management."""

    def test_create_session_token(self):
        """Test session token generation."""
        token = create_session_token()

        assert len(token) >= 32

    def test_session_tokens_unique(self):
        """Test session tokens are unique."""
        tokens = {create_session_token() for _ in range(100)}

        assert len(tokens) == 100


class TestPasswordValidation:
    """Tests for password strength validation."""

    def test_password_too_short(self):
        """Test short password is rejected."""
        assert validate_password_strength("Short1") is False

    def test_password_no_uppercase(self):
        """Test password without uppercase is rejected."""
        assert validate_password_strength("lowercase123") is False

    def test_password_no_lowercase(self):
        """Test password without lowercase is rejected."""
        assert validate_password_strength("UPPERCASE123") is False

    def test_password_no_digit(self):
        """Test password without digit is rejected."""
        assert validate_password_strength("NoDigitsHere") is False

    def test_valid_password(self):
        """Test valid password is accepted."""
        assert validate_password_strength("SecurePass123") is True
