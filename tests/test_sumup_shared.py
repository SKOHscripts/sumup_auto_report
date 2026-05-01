"""Tests unitaires pour utils/sumup_shared.py."""
from datetime import date, datetime

import pytest

from utils.sumup_shared import (
    iso_week_label,
    normalize,
    parse_dt,
    remove_accents,
    safe_float,
    week_start,
)


# ── remove_accents ────────────────────────────────────────────────────────────

class TestRemoveAccents:
    """Tests de remove_accents."""

    def test_simple_accent(self):
        assert remove_accents("é") == "e"

    def test_multiple_accents(self):
        assert remove_accents("café") == "cafe"

    def test_empty_string(self):
        assert remove_accents("") == ""

    def test_no_accents_unchanged(self):
        assert remove_accents("hello") == "hello"

    def test_uppercase_accent(self):
        assert remove_accents("É") == "E"

    def test_full_sentence(self):
        assert remove_accents("Café en grains 1 kg") == "Cafe en grains 1 kg"

    def test_multiple_diacritic_types(self):
        assert remove_accents("àâäèêëîïôùûü") == "aaaeeeiiouuu"

    def test_cedilla(self):
        assert remove_accents("ç") == "c"


# ── normalize ─────────────────────────────────────────────────────────────────

class TestNormalize:
    """Tests de normalize."""

    def test_lowercase(self):
        assert normalize("HELLO") == "hello"

    def test_strips_leading_trailing_spaces(self):
        assert normalize("  hello  ") == "hello"

    def test_removes_accents(self):
        assert normalize("Café") == "cafe"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_combined_transformations(self):
        assert normalize("  Café en Grains  ") == "cafe en grains"

    def test_none_like_input(self):
        assert normalize(None) == ""

    def test_mixed_case_with_spaces(self):
        assert normalize("  Thé Vert  ") == "the vert"


# ── iso_week_label ────────────────────────────────────────────────────────────

class TestIsoWeekLabel:
    """Tests de iso_week_label."""

    def test_format_contains_w(self):
        label = iso_week_label(datetime(2026, 3, 2))
        assert "-W" in label

    def test_year_extracted_correctly(self):
        label = iso_week_label(datetime(2026, 6, 1))
        assert label.startswith("2026")

    def test_week_number_zero_padded(self):
        label = iso_week_label(datetime(2026, 1, 5))
        week_part = label.split("-W")[1]
        assert len(week_part) == 2

    def test_known_date_week(self):
        # 2026-04-20 est en semaine 17
        label = iso_week_label(datetime(2026, 4, 20))
        assert label == "2026-W17"

    def test_end_of_year_iso(self):
        # 2025-12-29 appartient à la semaine 1 de 2026 selon ISO
        label = iso_week_label(datetime(2025, 12, 29))
        assert label == "2026-W01"

    def test_different_weeks_differ(self):
        label1 = iso_week_label(datetime(2026, 4, 6))
        label2 = iso_week_label(datetime(2026, 4, 13))
        assert label1 != label2


# ── week_start ────────────────────────────────────────────────────────────────

class TestWeekStart:
    """Tests de week_start."""

    def test_returns_date_type(self):
        result = week_start(2026, 13)
        assert isinstance(result, date)

    def test_returns_monday(self):
        result = week_start(2026, 13)
        assert result.weekday() == 0

    def test_consecutive_weeks_differ_by_7(self):
        d1 = week_start(2026, 10)
        d2 = week_start(2026, 11)
        assert (d2 - d1).days == 7

    def test_consistency_with_iso_week_label(self):
        d = week_start(2026, 5)
        label = iso_week_label(datetime(d.year, d.month, d.day))
        assert "W05" in label

    def test_week_1_is_in_january_or_december(self):
        d = week_start(2026, 1)
        assert d.month in (1, 12)

    def test_week_52_is_in_december(self):
        d = week_start(2026, 52)
        assert d.month == 12


# ── safe_float ────────────────────────────────────────────────────────────────

class TestSafeFloat:
    """Tests de safe_float."""

    def test_integer_input(self):
        assert safe_float(42) == 42.0

    def test_string_number(self):
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0

    def test_empty_string_returns_default(self):
        assert safe_float("") == 0.0

    def test_invalid_string_returns_default(self):
        assert safe_float("not_a_number") == 0.0

    def test_custom_default(self):
        assert safe_float(None, default=99.0) == 99.0

    def test_zero_input(self):
        assert safe_float(0) == 0.0

    def test_float_input_unchanged(self):
        assert safe_float(1.5) == 1.5

    def test_returns_float_type(self):
        result = safe_float(5)
        assert isinstance(result, float)

    def test_negative_value(self):
        assert safe_float(-3.5) == -3.5


# ── parse_dt ──────────────────────────────────────────────────────────────────

class TestParseDt:
    """Tests de parse_dt."""

    def test_basic_iso_string(self):
        dt = parse_dt("2026-04-20T14:30:00")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 20

    def test_z_suffix_accepted(self):
        dt = parse_dt("2026-04-20T14:30:00Z")
        assert dt is not None

    def test_empty_string_returns_none(self):
        assert parse_dt("") is None

    def test_none_input_returns_none(self):
        assert parse_dt(None) is None

    def test_invalid_string_returns_none(self):
        assert parse_dt("not-a-date") is None

    def test_returns_datetime_type(self):
        dt = parse_dt("2026-01-01T00:00:00")
        assert isinstance(dt, datetime)

    def test_time_components_preserved(self):
        dt = parse_dt("2026-06-15T08:45:30")
        assert dt.hour == 8
        assert dt.minute == 45
        assert dt.second == 30

    def test_timezone_offset(self):
        dt = parse_dt("2026-04-20T10:00:00+02:00")
        assert dt is not None
