"""Tests unitaires pour utils/mail_utils.py."""
import io
import os

import pytest

from utils.mail_utils import (
    _parse_email_list,
    build_log_footer,
    get_email_settings,
    load_project_env,
    resolve_recipients,
)


# ── _parse_email_list ─────────────────────────────────────────────────────────

class TestParseEmailList:
    """Tests de _parse_email_list."""

    def test_single_email(self):
        assert _parse_email_list("a@b.com") == ["a@b.com"]

    def test_multiple_emails_comma_separated(self):
        assert _parse_email_list("a@b.com, c@d.com") == ["a@b.com", "c@d.com"]

    def test_empty_string_returns_empty_list(self):
        assert _parse_email_list("") == []

    def test_none_input_returns_empty_list(self):
        assert _parse_email_list(None) == []

    def test_strips_whitespace_around_entries(self):
        assert _parse_email_list("  a@b.com  ,  c@d.com  ") == ["a@b.com", "c@d.com"]

    def test_filters_empty_entries(self):
        assert _parse_email_list(",a@b.com,,c@d.com,") == ["a@b.com", "c@d.com"]

    def test_single_entry_with_spaces(self):
        assert _parse_email_list("  hello@world.org  ") == ["hello@world.org"]


# ── load_project_env ──────────────────────────────────────────────────────────

class TestLoadProjectEnv:
    """Tests de load_project_env."""

    def test_no_required_vars_does_not_raise(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=hello\n", encoding="utf-8")
        result = load_project_env(env_file=str(env_file), required_vars=[])
        assert result is not None

    def test_missing_required_var_raises_runtime_error(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        monkeypatch.delenv("_SUMUP_TEST_MISSING_VAR_", raising=False)
        with pytest.raises(RuntimeError, match="_SUMUP_TEST_MISSING_VAR_"):
            load_project_env(env_file=str(env_file), required_vars=["_SUMUP_TEST_MISSING_VAR_"])

    def test_nonexistent_env_file_silently_skipped(self, tmp_path):
        fake_path = tmp_path / "nonexistent.env"
        result = load_project_env(env_file=str(fake_path), required_vars=[])
        assert result is not None

    def test_env_var_loaded_from_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("_SUMUP_TEST_LOADED_VAR_", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("_SUMUP_TEST_LOADED_VAR_=hello_from_env\n", encoding="utf-8")
        load_project_env(env_file=str(env_file), required_vars=[])
        assert os.getenv("_SUMUP_TEST_LOADED_VAR_") == "hello_from_env"

    def test_returns_path_object(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        result = load_project_env(env_file=str(env_file), required_vars=[])
        from pathlib import Path
        assert isinstance(result, Path)

    def test_multiple_missing_vars_all_listed(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        monkeypatch.delenv("_MISSING_A_", raising=False)
        monkeypatch.delenv("_MISSING_B_", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            load_project_env(
                env_file=str(env_file),
                required_vars=["_MISSING_A_", "_MISSING_B_"],
            )
        assert "_MISSING_A_" in str(exc_info.value)
        assert "_MISSING_B_" in str(exc_info.value)


# ── get_email_settings ────────────────────────────────────────────────────────

class TestGetEmailSettings:
    """Tests de get_email_settings."""

    def test_returns_dict_with_required_keys(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_PORT", "587")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASS", "secret")
        monkeypatch.setenv("EMAIL_TO", "dest@test.com")
        settings = get_email_settings()
        for key in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "EMAIL_TO_LIST", "MAILING_LISTS"):
            assert key in settings

    def test_default_smtp_host_when_unset(self, monkeypatch):
        monkeypatch.delenv("SMTP_HOST", raising=False)
        settings = get_email_settings()
        assert settings["SMTP_HOST"] == "smtp.gmail.com"

    def test_smtp_port_cast_to_int(self, monkeypatch):
        monkeypatch.setenv("SMTP_PORT", "465")
        settings = get_email_settings()
        assert settings["SMTP_PORT"] == 465
        assert isinstance(settings["SMTP_PORT"], int)

    def test_mailing_lists_contains_all_keys(self, monkeypatch):
        monkeypatch.setenv("EMAIL_TO", "a@b.com")
        settings = get_email_settings()
        for key in ("default", "all_ca", "finance", "vie"):
            assert key in settings["MAILING_LISTS"]

    def test_email_to_list_parsed(self, monkeypatch):
        monkeypatch.setenv("EMAIL_TO", "x@y.com, z@w.com")
        settings = get_email_settings()
        assert "x@y.com" in settings["EMAIL_TO_LIST"]
        assert "z@w.com" in settings["EMAIL_TO_LIST"]

    def test_email_from_defaults_to_smtp_user(self, monkeypatch):
        monkeypatch.setenv("SMTP_USER", "sender@test.com")
        monkeypatch.delenv("EMAIL_FROM", raising=False)
        settings = get_email_settings()
        assert settings["EMAIL_FROM"] == "sender@test.com"


# ── resolve_recipients ────────────────────────────────────────────────────────

class TestResolveRecipients:
    """Tests de resolve_recipients."""

    @pytest.fixture
    def settings(self, monkeypatch):
        monkeypatch.setenv("EMAIL_TO", "default@test.com")
        monkeypatch.setenv("EMAIL_TO_SUMUP_ALL_CA", "ca@test.com")
        monkeypatch.setenv("EMAIL_TO_SUMUP_FINANCE", "finance@test.com")
        return get_email_settings()

    def test_direct_to_list_takes_priority(self, settings):
        result = resolve_recipients(to_list=["a@b.com"], settings=settings)
        assert result == ["a@b.com"]

    def test_mailing_list_key_resolved(self, settings):
        result = resolve_recipients(mailing_list="default", settings=settings)
        assert "default@test.com" in result

    def test_none_mailing_list_returns_default(self, settings):
        result = resolve_recipients(mailing_list=None, settings=settings)
        assert isinstance(result, list)

    def test_list_of_emails_returned_as_is(self, settings):
        emails = ["x@y.com", "z@w.com"]
        result = resolve_recipients(mailing_list=emails, settings=settings)
        assert result == emails

    def test_tuple_of_emails_converted_to_list(self, settings):
        result = resolve_recipients(mailing_list=("a@b.com",), settings=settings)
        assert isinstance(result, list)
        assert "a@b.com" in result

    def test_unknown_key_returns_empty_list(self, settings):
        result = resolve_recipients(mailing_list="nonexistent_key_xyz", settings=settings)
        assert result == []

    def test_invalid_type_raises_type_error(self, settings):
        with pytest.raises(TypeError):
            resolve_recipients(mailing_list=42, settings=settings)

    def test_finance_mailing_list(self, settings):
        result = resolve_recipients(mailing_list="finance", settings=settings)
        assert "finance@test.com" in result


# ── build_log_footer ──────────────────────────────────────────────────────────

class TestBuildLogFooter:
    """Tests de build_log_footer."""

    def test_none_input_returns_empty_string(self):
        assert build_log_footer(None) == ""

    def test_empty_buffer_returns_empty_string(self):
        buf = io.StringIO()
        assert build_log_footer(buf) == ""

    def test_buffer_content_is_returned(self):
        buf = io.StringIO()
        buf.write("log line 1\nlog line 2")
        result = build_log_footer(buf)
        assert "log line 1" in result
        assert "log line 2" in result

    def test_trailing_whitespace_stripped(self):
        buf = io.StringIO()
        buf.write("some log\n\n")
        result = build_log_footer(buf)
        assert not result.endswith("\n")

    def test_multiple_lines_preserved(self):
        buf = io.StringIO()
        lines = ["line A", "line B", "line C"]
        buf.write("\n".join(lines))
        result = build_log_footer(buf)
        for line in lines:
            assert line in result
