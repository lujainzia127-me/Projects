"""
Email alert. Fires when the risk score crosses a threshold.

NEVER hard-code credentials. Set them in .streamlit/secrets.toml (preferred for
Streamlit apps) or as OS environment variables - either works:
    ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_EMAIL, ALERT_PASSWORD, ALERT_TO
For Gmail, use an App Password, not your real password.
"""

import os
import smtplib
from email.message import EmailMessage

THRESHOLD = 50  # send an alert at "High" or above


def _get_setting(key: str, default: str = None):
    """Check Streamlit secrets first (st.secrets persists across runs via
    .streamlit/secrets.toml), then fall back to OS environment variables."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError, AttributeError):
        pass  # no secrets.toml present, or not running inside Streamlit
    return os.environ.get(key, default)


def send_alert(risk_score: float, risk_band: str, summary_line: str = "", recipient: str = None):
    if risk_score < THRESHOLD:
        return False  # below threshold, no alert

    host = _get_setting("ALERT_SMTP_HOST", "smtp.gmail.com")
    port = int(_get_setting("ALERT_SMTP_PORT", "587"))
    sender = _get_setting("ALERT_EMAIL")
    password = _get_setting("ALERT_PASSWORD")

    # Only fall back to ALERT_TO if no valid recipient was explicitly provided
    if not recipient or not recipient.strip():
        recipient = _get_setting("ALERT_TO")

    if not all([sender, password, recipient]):
        print(
            "Email not configured - add ALERT_EMAIL, ALERT_PASSWORD "
            "to .streamlit/secrets.toml (or set them as environment variables), "
            "and make sure a recipient email was provided."
        )
        return False

    

    msg = EmailMessage()
    msg["Subject"] = f"[{risk_band}] Network threat alert - risk {risk_score}"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        f"Risk score {risk_score} ({risk_band}).\n{summary_line}\n"
        "Please investigate immediately."
    )

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        return True
    except smtplib.SMTPAuthenticationError:
        print(
            "SMTP authentication failed - check ALERT_EMAIL/ALERT_PASSWORD. "
            "For Gmail this must be a 16-character App Password, not your login password."
        )
        return False
    except Exception as e:
        print(f"Failed to send alert email: {e}")
        return False