"""Tests unitaires pour stocks/gdrive_loader.py."""
import io
from unittest.mock import MagicMock, patch

import pytest

import stocks.gdrive_loader as gdrive_mod
from stocks.gdrive_loader import _check_google_available, extract_file_id_from_url


# ── extract_file_id_from_url ──────────────────────────────────────────────────

class TestExtractFileIdFromUrl:
    """Tests de extract_file_id_from_url."""

    def test_spreadsheet_url(self):
        url = "https://docs.google.com/spreadsheets/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ01/edit"
        assert extract_file_id_from_url(url) == "1aBcDeFgHiJkLmNoPqRsTuVwXyZ01"

    def test_drive_file_url(self):
        url = "https://drive.google.com/file/d/1xYzAbCdEfGhIjKlMnOpQrStUv12/view"
        assert extract_file_id_from_url(url) == "1xYzAbCdEfGhIjKlMnOpQrStUv12"

    def test_raw_id_returned_unchanged(self):
        raw = "a" * 25
        assert extract_file_id_from_url(raw) == raw

    def test_raw_id_minimum_length(self):
        raw = "a" * 20
        assert extract_file_id_from_url(raw) == raw

    def test_invalid_url_raises_value_error(self):
        with pytest.raises(ValueError, match="Impossible"):
            extract_file_id_from_url("https://example.com/not-a-drive-url")

    def test_short_string_with_slash_raises(self):
        with pytest.raises(ValueError):
            extract_file_id_from_url("too/short")

    def test_url_with_query_params(self):
        url = "https://drive.google.com/file/d/AbCdEfGhIjKlMnOpQrStUvWx12345/view?usp=sharing"
        assert extract_file_id_from_url(url) == "AbCdEfGhIjKlMnOpQrStUvWx12345"

    def test_id_stripped_of_whitespace(self):
        raw = "  " + "b" * 20 + "  "
        result = extract_file_id_from_url(raw.strip())
        assert result == "b" * 20

    def test_id_with_hyphens_and_underscores(self):
        url = "https://drive.google.com/file/d/1aB-cD_eF-gH_iJ-kL_mN-oP12/view"
        result = extract_file_id_from_url(url)
        assert result == "1aB-cD_eF-gH_iJ-kL_mN-oP12"


# ── _check_google_available ───────────────────────────────────────────────────

class TestCheckGoogleAvailable:
    """Tests de _check_google_available."""

    def test_raises_import_error_when_unavailable(self):
        original = gdrive_mod._GOOGLE_AVAILABLE
        try:
            gdrive_mod._GOOGLE_AVAILABLE = False
            with pytest.raises(ImportError, match="Google"):
                _check_google_available()
        finally:
            gdrive_mod._GOOGLE_AVAILABLE = original

    def test_no_exception_when_available(self):
        original = gdrive_mod._GOOGLE_AVAILABLE
        try:
            gdrive_mod._GOOGLE_AVAILABLE = True
            _check_google_available()
        finally:
            gdrive_mod._GOOGLE_AVAILABLE = original

    def test_error_message_mentions_install(self):
        original = gdrive_mod._GOOGLE_AVAILABLE
        try:
            gdrive_mod._GOOGLE_AVAILABLE = False
            with pytest.raises(ImportError, match="pip install"):
                _check_google_available()
        finally:
            gdrive_mod._GOOGLE_AVAILABLE = original


# ── _download_from_service ────────────────────────────────────────────────────

class TestDownloadFromService:
    """Tests de _download_from_service avec service mocké."""

    def _make_service(self, content: bytes = b"fake content") -> MagicMock:
        mock_service = MagicMock()
        mock_service.files().get_media.return_value = MagicMock()
        return mock_service

    def test_returns_bytes_on_success(self):
        from stocks.gdrive_loader import _download_from_service

        mock_downloader = MagicMock()
        mock_downloader.next_chunk.return_value = (None, True)
        mock_service = self._make_service()

        def side_effect(buf, req):
            buf.write(b"fake content")
            return mock_downloader

        with patch("stocks.gdrive_loader.MediaIoBaseDownload", side_effect=side_effect):
            result = _download_from_service(mock_service, "fake_id")

        assert isinstance(result, bytes)
        assert result == b"fake content"

    def test_raises_permission_error_on_403(self):
        from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
        from stocks.gdrive_loader import _download_from_service

        mock_resp = MagicMock()
        mock_resp.status = 403
        http_error = HttpError(mock_resp, b"Forbidden")

        mock_service = MagicMock()
        mock_service.files().get_media.side_effect = http_error

        with pytest.raises(PermissionError, match="Accès refusé"):
            _download_from_service(mock_service, "fake_id")

    def test_raises_file_not_found_on_404(self):
        from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
        from stocks.gdrive_loader import _download_from_service

        mock_resp = MagicMock()
        mock_resp.status = 404
        http_error = HttpError(mock_resp, b"Not Found")

        mock_service = MagicMock()
        mock_service.files().get_media.side_effect = http_error

        with pytest.raises(FileNotFoundError, match="introuvable"):
            _download_from_service(mock_service, "fake_id")

    def test_raises_runtime_error_on_other_http_error(self):
        from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
        from stocks.gdrive_loader import _download_from_service

        mock_resp = MagicMock()
        mock_resp.status = 500
        http_error = HttpError(mock_resp, b"Server Error")

        mock_service = MagicMock()
        mock_service.files().get_media.side_effect = http_error

        with pytest.raises(RuntimeError, match="HTTP 500"):
            _download_from_service(mock_service, "fake_id")


# ── download_file_as_bytes ────────────────────────────────────────────────────

class TestDownloadFileAsBytes:
    """Tests de download_file_as_bytes (intégration partielle avec mocks)."""

    def test_raises_file_not_found_for_missing_credentials(self, tmp_path):
        from stocks.gdrive_loader import download_file_as_bytes

        with pytest.raises(FileNotFoundError, match="credentials"):
            download_file_as_bytes("some_file_id", str(tmp_path / "nonexistent.json"))

    def test_raises_import_error_when_google_unavailable(self, tmp_path):
        from stocks.gdrive_loader import download_file_as_bytes

        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}", encoding="utf-8")
        original = gdrive_mod._GOOGLE_AVAILABLE
        try:
            gdrive_mod._GOOGLE_AVAILABLE = False
            with pytest.raises(ImportError):
                download_file_as_bytes("some_file_id", str(creds_file))
        finally:
            gdrive_mod._GOOGLE_AVAILABLE = original
