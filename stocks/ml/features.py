#!/usr/bin/env python3
"""Feature engineering pour la prévision de consommation hebdomadaire.

Pipeline standard appliqué au DataFrame produit par ``stocks.ml.dataset`` :

  1. ``add_calendar_features``     : mois, semaine de l'année, position dans le mois,
                                      flag jour férié FR, début/fin de mois
  2. ``add_lag_features``          : usage des semaines t-1, t-2, t-4, t-12
  3. ``add_rolling_features``      : moyennes / écarts-types mobiles 4 et 13 semaines

Toutes les transformations sont calculées **par SKU** et **strictement
historiques** : à la semaine t, seules les valeurs observées en t-1, t-2, …
servent à construire les features. Pas de fuite de futur.

La fonction ``prepare_training_table`` retourne ``(X, y, meta)`` où ``meta``
contient ``stock_sku`` et ``week_label`` pour la traçabilité (split temporel,
backtest).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

# ─── Jours fériés français ───────────────────────────────────────────────────


def _easter_sunday(year: int) -> date:
    """Calcule la date de Pâques pour une année donnée (algorithme de Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def fr_public_holidays(year: int) -> set[date]:
    """Retourne l'ensemble des jours fériés légaux français pour une année."""
    from datetime import timedelta  # pylint: disable=import-outside-toplevel

    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),                         # Jour de l'an
        easter + timedelta(days=1),               # Lundi de Pâques
        date(year, 5, 1),                         # Fête du Travail
        date(year, 5, 8),                         # Victoire 1945
        easter + timedelta(days=39),              # Ascension
        easter + timedelta(days=50),              # Lundi de Pentecôte
        date(year, 7, 14),                        # Fête nationale
        date(year, 8, 15),                        # Assomption
        date(year, 11, 1),                        # Toussaint
        date(year, 11, 11),                       # Armistice 1918
        date(year, 12, 25),                       # Noël
    }


def _holidays_in_week(week_start: date) -> int:
    """Nombre de jours fériés FR tombant dans la semaine commençant le lundi `week_start`."""
    from datetime import timedelta  # pylint: disable=import-outside-toplevel

    end = week_start + timedelta(days=6)
    holidays = fr_public_holidays(week_start.year)
    if end.year != week_start.year:
        holidays |= fr_public_holidays(end.year)
    return sum(1 for h in holidays if week_start <= h <= end)


# ─── Features calendrier ─────────────────────────────────────────────────────


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features calendrier dérivées de `week_start`.

    Colonnes ajoutées :
      - month, week_of_year, week_in_month
      - n_holidays : nb de jours fériés dans la semaine
      - is_first_week_of_month, is_last_week_of_month
      - sin_week, cos_week : encodage cyclique de la semaine (52)
    """
    out = df.copy()
    ws = out["week_start"].apply(_to_date_safe)
    out["month"] = ws.apply(lambda d: d.month).astype("int32")
    out["week_of_year"] = out["week"].astype("int32")
    out["week_in_month"] = ws.apply(lambda d: (d.day - 1) // 7 + 1).astype("int32")
    out["n_holidays"] = ws.apply(_holidays_in_week).astype("int32")
    out["is_first_week_of_month"] = (out["week_in_month"] == 1).astype("int32")
    out["is_last_week_of_month"] = ws.apply(_is_last_week_of_month).astype("int32")
    angle = 2 * np.pi * out["week_of_year"].astype("float64") / 52.0
    out["sin_week"] = np.sin(angle)
    out["cos_week"] = np.cos(angle)
    return out


def _to_date_safe(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return date.fromisoformat(str(value))


def _is_last_week_of_month(week_start_date: date) -> int:
    from datetime import timedelta  # pylint: disable=import-outside-toplevel

    next_week = week_start_date + timedelta(days=7)
    return int(next_week.month != week_start_date.month)


# ─── Lags & rolling ──────────────────────────────────────────────────────────


DEFAULT_LAGS = (1, 2, 4, 12)
DEFAULT_ROLLING_WINDOWS = (4, 13)


def add_lag_features(
    df: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    target_col: str = "usage",
) -> pd.DataFrame:
    """Ajoute les colonnes ``lag_<k>`` calculées par SKU sur l'historique trié."""
    out = df.sort_values(["stock_sku", "year", "week"]).copy()
    grouped = out.groupby("stock_sku", sort=False)[target_col]
    for k in lags:
        out[f"lag_{k}"] = grouped.shift(k)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    target_col: str = "usage",
) -> pd.DataFrame:
    """Ajoute moyennes et écart-types mobiles, calculés sur les valeurs **passées**.

    On décale d'une semaine pour que la valeur courante n'entre pas dans la
    fenêtre (pas de fuite de futur).
    """
    out = df.sort_values(["stock_sku", "year", "week"]).copy()
    shifted = out.groupby("stock_sku", sort=False)[target_col].shift(1)
    for w in windows:
        out[f"rolling_mean_{w}"] = (
            shifted.groupby(out["stock_sku"]).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        out[f"rolling_std_{w}"] = (
            shifted.groupby(out["stock_sku"]).rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        )
    return out


# ─── Construction de la table d'entraînement ─────────────────────────────────


def build_feature_table(
    df: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    target_col: str = "usage",
) -> pd.DataFrame:
    """Pipeline complet : calendrier + lags + rolling, conserve toutes les lignes.

    Les lignes initiales (sans assez d'historique) auront des NaN dans les
    lags / rolling — au choix de l'appelant de filtrer (cf. ``prepare_training_table``).
    """
    out = add_calendar_features(df)
    out = add_lag_features(out, lags=lags, target_col=target_col)
    out = add_rolling_features(out, windows=windows, target_col=target_col)
    return out


def prepare_training_table(
    df: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    target_col: str = "usage",
    drop_warmup: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construit ``(X, y, meta)`` à partir de l'historique brut.

    - ``X``    : DataFrame des features (numériques + ``stock_sku`` catégoriel).
    - ``y``    : série de la cible (``usage`` par défaut).
    - ``meta`` : DataFrame avec ``stock_sku``, ``week_label``, ``week_start`` pour
      tracer chaque ligne (utile pour split temporel / backtest).

    Si ``drop_warmup=True`` (défaut), les lignes pour lesquelles le plus grand
    lag n'est pas disponible sont retirées (warm-up nécessaire pour avoir tous
    les lags).
    """
    full = build_feature_table(df, lags=lags, windows=windows, target_col=target_col)
    if drop_warmup and lags:
        max_lag = max(lags)
        full = full.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)

    feature_cols = [
        "stock_sku",
        "month",
        "week_of_year",
        "week_in_month",
        "n_holidays",
        "is_first_week_of_month",
        "is_last_week_of_month",
        "sin_week",
        "cos_week",
        *[f"lag_{k}" for k in lags],
        *[f"rolling_mean_{w}" for w in windows],
        *[f"rolling_std_{w}" for w in windows],
    ]
    X = full[feature_cols].copy()
    X[[f"rolling_std_{w}" for w in windows]] = (
        X[[f"rolling_std_{w}" for w in windows]].fillna(0.0)
    )
    y = full[target_col].astype("float64")
    meta = full[["stock_sku", "week_label", "week_start", "year", "week"]].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta


__all__ = [
    "DEFAULT_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
    "add_calendar_features",
    "add_lag_features",
    "add_rolling_features",
    "build_feature_table",
    "fr_public_holidays",
    "prepare_training_table",
]
