"""Transactional email helpers for account provisioning."""
from email.message import EmailMessage
import os
import smtplib


class EmailDeliveryError(Exception):
    """Raised when the system cannot safely deliver a transactional email."""


def _settings() -> dict[str, object]:
    host = os.getenv("SMTP_HOST")
    sender = os.getenv("SMTP_FROM")
    if not host or not sender:
        raise EmailDeliveryError(
            "Email delivery is not configured. Set SMTP_HOST and SMTP_FROM before creating accounts."
        )
    return {
        "host": host,
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD"),
        "sender": sender,
        "starttls": os.getenv("SMTP_STARTTLS", "true").lower() in {"1", "true", "yes"},
    }


def send_account_credentials(*, recipient: str, full_name: str, role: str, temporary_password: str) -> None:
    """Send the one-time credentials for an administrator or PM account."""
    settings = _settings()
    login_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:5173').rstrip('/')}/login"
    role_label = "Organisation Administrator" if role == "org_admin" else "Project Manager"

    message = EmailMessage()
    message["Subject"] = "Your PHPS account has been created"
    message["From"] = settings["sender"]
    message["To"] = recipient
    message.set_content(
        f"Hello {full_name},\n\n"
        f"A {role_label} account has been created for you in the Project Health Prediction System (PHPS).\n\n"
        f"Login URL: {login_url}\n"
        f"Email: {recipient}\n"
        f"Temporary password: {temporary_password}\n\n"
        "For security, you will be required to change this temporary password when you first sign in. "
        "Do not share these credentials with anyone.\n\n"
        "PHPS Team\n"
    )

    try:
        with smtplib.SMTP(settings["host"], settings["port"], timeout=15) as server:
            server.ehlo()
            if settings["starttls"]:
                server.starttls()
                server.ehlo()
            if settings["username"]:
                if not settings["password"]:
                    raise EmailDeliveryError("SMTP_USERNAME is set but SMTP_PASSWORD is missing.")
                server.login(settings["username"], settings["password"])
            server.send_message(message)
    except EmailDeliveryError:
        raise
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailDeliveryError(f"Could not send the account email: {exc}") from exc
