"""
AES-256 encryption/decryption using Fernet (symmetric key).
Used to encrypt Jira and Gmail API tokens before storing in the database.

GDPR Note: Raw tokens are encrypted at rest using AES-256.
The key is stored only in the .env file, never in the database.
"""
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException
import os


def _get_fernet() -> Fernet:
    """Load the Fernet key from environment and return a Fernet instance."""
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY is not set in your .env file. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_token(plain_text: str) -> str:
    """
    Encrypt a plaintext string (e.g. a Jira API token).
    Returns the encrypted bytes as a UTF-8 string safe for database storage.
    """
    if not plain_text:
        return ""
    fernet = _get_fernet()
    encrypted_bytes = fernet.encrypt(plain_text.encode("utf-8"))
    return encrypted_bytes.decode("utf-8")


def decrypt_token(encrypted_text: str) -> str:
    """
    Decrypt a previously encrypted string.
    Raises HTTP 500 if the key has changed or the data is corrupted.
    """
    if not encrypted_text:
        return ""
    try:
        fernet = _get_fernet()
        decrypted_bytes = fernet.decrypt(encrypted_text.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except InvalidToken:
        raise HTTPException(
            status_code=500,
            detail="Failed to decrypt token. "
                   "This may happen if ENCRYPTION_KEY was rotated.",
        )
