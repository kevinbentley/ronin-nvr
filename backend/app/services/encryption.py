"""Encryption service for camera credentials."""

import secrets
import string
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


def generate_encryption_key() -> str:
    """Generate a new Fernet encryption key."""
    return Fernet.generate_key().decode()


def encrypt_password(password: str) -> str:
    """Encrypt a password using Fernet symmetric encryption.

    Returns the encrypted password as a base64 string.
    If encryption_key is not configured, returns the password as-is (for dev mode).
    """
    settings = get_settings()

    if not settings.encryption_key:
        # No encryption key configured - return as-is (dev mode)
        return password

    try:
        fernet = Fernet(settings.encryption_key.encode())
        encrypted = fernet.encrypt(password.encode())
        return encrypted.decode()
    except Exception:
        # If encryption fails, return as-is
        return password


def decrypt_password(encrypted: str) -> str:
    """Decrypt a password encrypted with Fernet.

    Returns the decrypted password.
    If decryption fails (not encrypted or wrong key), returns the value as-is.
    """
    settings = get_settings()

    if not settings.encryption_key:
        # No encryption key configured - return as-is
        return encrypted

    try:
        fernet = Fernet(settings.encryption_key.encode())
        decrypted = fernet.decrypt(encrypted.encode())
        return decrypted.decode()
    except (InvalidToken, Exception):
        # If decryption fails, the value is likely not encrypted
        return encrypted


def generate_random_password(length: int = 16) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    # Ensure we have at least one of each character type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(string.punctuation),
    ]
    # Fill the rest with random choices
    password.extend(secrets.choice(alphabet) for _ in range(length - 4))
    # Shuffle to avoid predictable positions
    secrets.SystemRandom().shuffle(password)
    return "".join(password)
