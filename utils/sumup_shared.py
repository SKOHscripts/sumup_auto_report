#!/usr/bin/env python3
"""
Fonctions utilitaires partagées entre les scripts SumUp (stocks, statistics, adhesions).

Importez ces fonctions plutôt que de les redéfinir localement :
  from utils.sumup_shared import remove_accents, normalize, iso_week_label, week_start
  from utils.sumup_shared import safe_float, parse_dt
"""

import unicodedata
from datetime import datetime, timedelta, date
from typing import Any, Optional


def remove_accents(text: str) -> str:
    if not text:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return remove_accents(text or "").strip().lower()


def iso_week_label(dt: datetime) -> str:
    """Retourne le label ISO de la semaine, ex: '2026-W13'."""
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def week_start(year: int, week: int) -> date:
    """Retourne le lundi de la semaine ISO donnée."""
    jan4 = date(year, 1, 4)
    start = jan4 - timedelta(days=jan4.isoweekday() - 1)
    return start + timedelta(weeks=week - 1)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
