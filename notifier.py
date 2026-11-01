"""Email notifications for ReClaim lost-item matches."""

import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Dict, Tuple

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _smtp_user() -> str:
    """Accept both names so existing Cloud secrets continue to work."""
    return (os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER") or "").strip()


def valid_email(address: str) -> bool:
    return bool(EMAIL_RE.fullmatch((address or "").strip()))


def smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and _smtp_user() and os.getenv("SMTP_PASSWORD"))


def build_message(found_row: Dict[str, Any], confidence: float) -> Tuple[str, str]:
    subject = "ReClaim: a found item may match your lost item"
    body = (
        "Hello,\n\n"
        "A found-item report that may match your lost item was submitted on ReClaim.\n\n"
        f"Description: {found_row.get('description', 'Not provided')}\n"
        f"Found at: {found_row.get('location', 'Not provided')}\n"
        f"Date: {found_row.get('date', 'Not provided')}\n"
        f"Match score: {confidence:.0%} (potential match only)\n\n"
        "This is an automated suggestion and does not confirm ownership. Open ReClaim, "
        "review the report, and click 'This is my item' to confirm.\n\n"
        "ReClaim"
    )
    return subject, body


def send_email(to: str, subject: str, body: str) -> Tuple[str, str]:
    """Send one email and return ``(status, detail)`` for the app notification log."""
    if not valid_email(to):
        return "no_email", "contact on the lost report is not a valid email address"
    if not smtp_configured():
        return "failed", "SMTP is not configured; add SMTP_HOST, SMTP_USERNAME, and SMTP_PASSWORD to Streamlit Secrets"

    user = _smtp_user()
    host = os.environ["SMTP_HOST"]
    password = os.environ["SMTP_PASSWORD"]
    port = int(os.getenv("SMTP_PORT", "587"))

    message = EmailMessage()
    message["From"] = (os.getenv("SMTP_FROM") or user).strip()
    message["To"] = to.strip()
    message["Subject"] = subject
    message.set_content(body)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20, context=ssl.create_default_context()) as server:
                server.login(user, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(user, password)
                server.send_message(message)
        return "sent", "email accepted by the SMTP server"
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        return "failed", f"SMTP error: {exc}"[:180]
