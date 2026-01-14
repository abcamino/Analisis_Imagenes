"""Security utilities for authentication."""

import secrets
import re
import hashlib
import os


def _sha256_hash(password: str, salt: bytes = None) -> str:
    """Simple SHA256-based password hashing (for development/testing)."""
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt.hex() + ':' + key.hex()


def _sha256_verify(password: str, hashed: str) -> bool:
    """Verify SHA256 hash."""
    try:
        salt_hex, key_hex = hashed.split(':')
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        actual_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return actual_key == expected_key
    except (ValueError, AttributeError):
        return False


def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2-SHA256.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return _sha256_hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored hash to check against

    Returns:
        True if password matches, False otherwise
    """
    return _sha256_verify(plain_password, hashed_password)


def create_session_token() -> str:
    """
    Generate a secure random session token.

    Returns:
        URL-safe random token string
    """
    return secrets.token_urlsafe(32)


def validate_password_strength(password: str) -> bool:
    """
    Validate password meets minimum requirements.

    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit

    Args:
        password: Password to validate

    Returns:
        True if password is strong enough
    """
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    return True
