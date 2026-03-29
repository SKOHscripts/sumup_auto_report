# mail_utils.py
import io
import os
import ssl
import smtplib
import logging
from pathlib import Path
from email.message import EmailMessage

from dotenv import load_dotenv


def load_project_env(env_file=None, required_vars=None, logger=None):
    logger = logger or logging.getLogger(__name__)

    if env_file is None:
        env_file = Path(__file__).resolve().parent / ".env"
    else:
        env_file = Path(env_file)

    if not env_file.exists():
        raise RuntimeError(f".env introuvable : {env_file}")

    load_dotenv(dotenv_path=env_file, override=True)
    logger.info(f".env chargé depuis : {env_file.name}")

    missing = [var for var in (required_vars or []) if not os.getenv(var)]

    if missing:
        raise RuntimeError(f"Variables manquantes dans .env : {', '.join(missing)}")

    return env_file


def setup_memory_log_capture(datefmt="%Y-%m-%d %H:%M:%S"):
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt=datefmt
        ))
    logging.getLogger().addHandler(handler)

    return buffer, handler


def get_email_settings():
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    email_from = os.getenv("EMAIL_FROM", smtp_user)
    email_to_list = [e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()]

    return {
        "SMTP_HOST": smtp_host,
        "SMTP_PORT": smtp_port,
        "SMTP_USER": smtp_user,
        "SMTP_PASS": smtp_pass,
        "EMAIL_FROM": email_from,
        "EMAIL_TO_LIST": email_to_list,
        }


def send_email(
    subject,
    body,
    attachments=None,
    to_list=None,
    from_addr=None,
    logger=None,
        ):
    logger = logger or logging.getLogger(__name__)
    settings = get_email_settings()

    recipients = to_list or settings["EMAIL_TO_LIST"]
    sender = from_addr or settings["EMAIL_FROM"]

    if not recipients:
        logger.warning("Aucun destinataire défini (EMAIL_TO vide). Email non envoyé.")

        return False

    if not settings["SMTP_USER"] or not settings["SMTP_PASS"]:
        logger.warning("SMTP_USER ou SMTP_PASS manquant. Email non envoyé.")

        return False

    em = EmailMessage()
    em["From"] = sender
    em["To"] = ", ".join(recipients)
    em["Subject"] = subject
    em.set_content(body)

    for attachment in attachments or []:
        path = Path(attachment)

        if not path.exists():
            logger.warning(f"Pièce jointe introuvable, ignorée : {path}")

            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            maintype, subtype = "application", "pdf"
        elif suffix == ".png":
            maintype, subtype = "image", "png"
        elif suffix == ".jpg" or suffix == ".jpeg":
            maintype, subtype = "image", "jpeg"
        elif suffix == ".csv":
            maintype, subtype = "text", "csv"
        elif suffix == ".json":
            maintype, subtype = "application", "json"
        else:
            maintype, subtype = "application", "octet-stream"

        with open(path, "rb") as f:
            em.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name
                )

    ctx = ssl.create_default_context()
    with smtplib.SMTP(settings["SMTP_HOST"], settings["SMTP_PORT"], timeout=20) as srv:
        srv.ehlo()
        srv.starttls(context=ctx)
        srv.ehlo()
        srv.login(settings["SMTP_USER"], settings["SMTP_PASS"])
        srv.send_message(em)

    logger.info(f"Email envoyé à : {', '.join(recipients)}")

    return True


def build_log_footer(log_buffer):
    if not log_buffer:
        return ""

    return log_buffer.getvalue().strip()
