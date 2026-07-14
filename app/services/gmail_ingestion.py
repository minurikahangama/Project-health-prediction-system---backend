"""Read project-related messages from one monitored Gmail inbox via IMAP."""
from __future__ import annotations

from datetime import date, timedelta
from email import message_from_bytes
from email.header import decode_header, make_header
import imaplib
import os
from typing import Dict, List

from app.utils.encryption import decrypt_token


class GmailIngestionError(RuntimeError):
    """Raised when the monitored Gmail inbox cannot be read safely."""


def _settings(account_email: str | None = None, encrypted_token: str | None = None) -> Dict[str, object]:
    """Return project-specific IMAP credentials, with legacy env fallback."""
    username = account_email or os.getenv("GMAIL_IMAP_USERNAME")
    password = decrypt_token(encrypted_token) if encrypted_token else os.getenv("GMAIL_IMAP_PASSWORD")
    if not username or not password:
        raise GmailIngestionError("Gmail polling is not configured. Set GMAIL_IMAP_USERNAME and GMAIL_IMAP_PASSWORD in .env.")
    return {
        "host": os.getenv("GMAIL_IMAP_HOST", "imap.gmail.com"),
        "port": int(os.getenv("GMAIL_IMAP_PORT", "993")),
        "username": username,
        "password": password,
    }


def _decoded(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _plain_text(message) -> str:
    """Extract plain text without retaining attachments or HTML-only markup."""
    parts = message.walk() if message.is_multipart() else [message]
    text_parts: List[str] = []
    for part in parts:
        if part.get_content_type() != "text/plain" or part.get_content_disposition() == "attachment":
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            text_parts.append(payload.decode(charset, errors="replace"))
        except LookupError:
            text_parts.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(text_parts).strip()


def validate_imap_credentials(account_email: str, app_password: str) -> None:
    """Verify a Gmail IMAP app password without persisting it."""
    if not account_email or not app_password:
        raise GmailIngestionError("Gmail account email and app password are required")
    try:
        with imaplib.IMAP4_SSL(os.getenv("GMAIL_IMAP_HOST", "imap.gmail.com"), int(os.getenv("GMAIL_IMAP_PORT", "993"))) as mailbox:
            mailbox.login(account_email, app_password)
    except (OSError, imaplib.IMAP4.error) as exc:
        raise GmailIngestionError("Gmail authorization failed. Check the account email and app password.") from exc


def fetch_messages(*, sender: str, account_email: str | None = None, encrypted_token: str | None = None, lookback_days: int = 7) -> List[dict]:
    """Return recent messages to or from the monitored address.

    The caller filters by project identifier and records message IDs, so a
    seven-day lookback safely catches up after temporary downtime.
    """
    settings = _settings(account_email, encrypted_token)
    since = (date.today() - timedelta(days=lookback_days)).strftime("%d-%b-%Y")
    try:
        with imaplib.IMAP4_SSL(settings["host"], settings["port"]) as mailbox:
            mailbox.login(settings["username"], settings["password"])
            # All Mail includes both incoming client messages and messages sent
            # by the project team.  Fall back to INBOX for non-Gmail IMAP hosts.
            status, _ = mailbox.select('"[Gmail]/All Mail"', readonly=True)
            if status != "OK":
                status, _ = mailbox.select("INBOX", readonly=True)
            if status != "OK":
                raise GmailIngestionError("Could not open the monitored mailbox")
            status, data = mailbox.search(
                None, "OR", "FROM", f'"{sender}"', "TO", f'"{sender}"', "SINCE", since
            )
            if status != "OK":
                raise GmailIngestionError("Could not search the Gmail inbox")
            ids = data[0].split()[-200:]
            messages: List[dict] = []
            for uid in reversed(ids):
                status, payload = mailbox.fetch(uid, "(RFC822)")
                if status != "OK" or not payload or not isinstance(payload[0], tuple):
                    continue
                message = message_from_bytes(payload[0][1])
                body = _plain_text(message)
                subject = _decoded(message.get("Subject"))
                if not body and not subject:
                    continue
                messages.append({
                    "id": message.get("Message-ID") or f"imap-{uid.decode(errors='replace')}",
                    "subject": subject,
                    "body": body,
                    "sender": _decoded(message.get("From")),
                    "received_at": _decoded(message.get("Date")),
                })
            return messages
    except GmailIngestionError:
        raise
    except (OSError, imaplib.IMAP4.error) as exc:
        raise GmailIngestionError(f"Gmail polling failed: {exc}") from exc
