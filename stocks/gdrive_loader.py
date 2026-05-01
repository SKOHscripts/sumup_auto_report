#!/usr/bin/env python3
"""Téléchargement d'un fichier depuis Google Drive via l'API v3 (Service Account)."""

import io
import logging
import os
import re

log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

try:
    from google.oauth2 import service_account  # type: ignore[import-untyped]
    from googleapiclient.discovery import build  # type: ignore[import-untyped]
    from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False


def _build_drive_service(creds):
    """Construit le service Google Drive v3 à partir de credentials déjà créés."""
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _download_from_service(service, file_id: str) -> bytes:
    """Télécharge un fichier Drive via le service fourni et retourne les octets."""
    try:
        request = service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        size = buffer.tell()
        log.info("Fichier Drive '%s' téléchargé (%d octets)", file_id, size)
        return buffer.getvalue()
    except HttpError as exc:
        status = exc.resp.status
        if status == 403:
            raise PermissionError(
                f"Accès refusé au fichier Drive '{file_id}'. "
                "Vérifiez que le fichier est partagé avec l'email du service account."
            ) from exc
        if status == 404:
            raise FileNotFoundError(
                f"Fichier Drive introuvable : '{file_id}'. Vérifiez GDRIVE_PURCHASES_FILE_ID."
            ) from exc
        raise RuntimeError(f"Erreur API Google Drive (HTTP {status}) : {exc}") from exc


def _check_google_available():
    if not _GOOGLE_AVAILABLE:
        raise ImportError(
            "Modules Google requis manquants. "
            "Installez : pip install google-api-python-client google-auth"
        )


def download_file_as_bytes(file_id: str, credentials_path: str) -> bytes:
    """Télécharge un fichier depuis Google Drive et retourne son contenu brut.

    Args:
        file_id: ID du fichier dans Google Drive (chaîne entre /d/ et / dans l'URL).
        credentials_path: Chemin absolu vers le JSON du service account Google.

    Raises:
        ImportError: Si google-api-python-client / google-auth ne sont pas installés.
        FileNotFoundError: Si credentials_path est introuvable ou fichier Drive absent.
        PermissionError: Si le service account n'a pas accès au fichier.
        RuntimeError: Pour toute autre erreur d'API Google Drive.
    """
    _check_google_available()
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Fichier credentials introuvable : {credentials_path}")
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=_SCOPES
    )
    return _download_from_service(_build_drive_service(creds), file_id)


def download_file_as_bytes_from_info(file_id: str, service_account_info: dict) -> bytes:
    """Télécharge un fichier Drive en passant les credentials directement en dict.

    Utilisé depuis Streamlit où les secrets sont stockés dans secrets.toml
    sans fichier JSON sur le disque.

    Args:
        file_id: ID du fichier dans Google Drive.
        service_account_info: Contenu du JSON service account sous forme de dict
            (clés : type, project_id, private_key, client_email, …).

    Raises:
        ImportError: Si google-api-python-client / google-auth ne sont pas installés.
        PermissionError: Si le service account n'a pas accès au fichier.
        RuntimeError: Pour toute autre erreur d'API Google Drive.
    """
    _check_google_available()
    creds = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=_SCOPES
    )
    return _download_from_service(_build_drive_service(creds), file_id)


def extract_file_id_from_url(url: str) -> str:
    """Extrait l'ID de fichier depuis une URL Google Drive ou Sheets.

    Formats acceptés :
      https://drive.google.com/file/d/FILE_ID/view
      https://docs.google.com/spreadsheets/d/FILE_ID/edit
      FILE_ID  (déjà un identifiant brut)
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]{20,})", url)
    if match:
        return match.group(1)
    if "/" not in url and len(url) >= 20:
        return url.strip()
    raise ValueError(
        f"Impossible d'extraire un ID de fichier depuis : '{url}'. "
        "Format attendu : https://drive.google.com/file/d/FILE_ID/view"
    )
