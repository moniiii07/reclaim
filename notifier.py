"""E-mail notification of lost-item owners (report section 3.1: the e-mail address
from the lost item report is used, so the finder never enters it)."""
from __future__ import annotations

import os
import re
import smtplib
from email.message import EmailMessage

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def valid_email(addr: str) -> bool:
    return bool(EMAIL_RE.match((addr or "").strip()))


def smtp_configured() -> bool:
    return all(os.getenv(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def build_message(found_row: dict, confidence: float) -> tuple[str, str]:
    subject = "ReClaim: a found item may match your lost item"
    body = (
        "Hello,\n\n"
        "A found-item report that may match your lost item was just submitted on ReClaim.\n\n"
        f"  Description : {found_row['description']}\n"
        f"  Found at    : {found_row['location']}\n"
        f"  Date        : {found_row['date']}\n"
        f"  Match score : {confidence:.0%} (potential match only)\n\n"
        "This is an automated suggestion and does not confirm ownership. Open ReClaim, review "
        "the report and click 'This is my item' to confirm. The finder's contact details are "
        "shown to you only after you confirm.\n\n- ReClaim")
    return subject, body


def send_email(to: str, subject: str, body: str) -> tuple[str, str]:
    """Return (status, detail): status is 'sent', 'failed', 'simulated' or 'no_email'."""
    if not valid_email(to):
        return "no_email", "contact on the lost report is not an e-mail address"
    if not smtp_configured():
        return "simulated", "SMTP not configured - notification logged only"
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM") or os.environ["SMTP_USER"]
    msg["To"], msg["Subject"] = to.strip(), subject
    msg.set_content(body)
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        if port == 465:
            with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], port, timeout=15) as s:
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=15) as s:
                s.starttls()
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                s.send_message(msg)
        return "sent", "e-mail delivered to SMTP server"
    except Exception as exc:  # network / auth errors
        return "failed", str(exc)[:120]