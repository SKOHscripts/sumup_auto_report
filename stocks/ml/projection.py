#!/usr/bin/env python3
"""Projection probabiliste de la consommation et de la date de rupture.

Workflow :

    1. ``forecast_horizon(model, history, horizon_weeks)``
       Prédit (q10, q50, q90) pour les `horizon` prochaines semaines en
       construisant les features de manière itérative : à chaque pas, le q50
       prédit devient le ``lag_1`` du pas suivant. Les rolling features sont
       recalculées sur la trajectoire q50.

    2. ``simulate_rupture(stock_initial, weekly_quantiles, ...)``
       Tire ``n_simulations`` trajectoires en samplant indépendamment dans
       l'intervalle [q10, q90] chaque semaine, puis cumule la décroissance
       du stock. Retourne un dict avec les percentiles P10 / P50 / P90 de la
       date de rupture.

Tout est conçu pour fonctionner par SKU : on appelle ces fonctions sur
l'historique d'un seul SKU à la fois.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from stocks.ml.features import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    add_calendar_features,
)


def _next_iso_week(year: int, week: int) -> tuple[int, int]:
    """Retourne (année, semaine) ISO de la semaine suivante."""
    from utils.sumup_shared import week_start  # pylint: disable=import-outside-toplevel

    monday = week_start(year, week) + timedelta(days=7)
    iso = monday.isocalendar()
    return iso.year, iso.week


def _row_features(
    sku: str,
    year: int,
    week: int,
    week_start_dt: date,
    lag_values: dict[int, float],
    rolling_history: list[float],
    lags: tuple[int, ...],
    windows: tuple[int, ...],
) -> dict:
    """Construit le vecteur de features pour une semaine future donnée."""
    base = pd.DataFrame([{
        "stock_sku": sku,
        "week_label": f"{year}-W{week:02d}",
        "year": year,
        "week": week,
        "week_start": week_start_dt,
        "usage": 0.0,
    }])
    cal = add_calendar_features(base).iloc[0].to_dict()
    feats = {
        "stock_sku": sku,
        "month": cal["month"],
        "week_of_year": cal["week_of_year"],
        "week_in_month": cal["week_in_month"],
        "n_holidays": cal["n_holidays"],
        "is_first_week_of_month": cal["is_first_week_of_month"],
        "is_last_week_of_month": cal["is_last_week_of_month"],
        "sin_week": cal["sin_week"],
        "cos_week": cal["cos_week"],
    }
    for k in lags:
        feats[f"lag_{k}"] = lag_values.get(k, 0.0)
    for w in windows:
        recent = rolling_history[-w:] if len(rolling_history) >= 1 else [0.0]
        feats[f"rolling_mean_{w}"] = float(np.mean(recent))
        feats[f"rolling_std_{w}"] = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
    return feats


def forecast_horizon(
    model,
    history_df: pd.DataFrame,
    horizon_weeks: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Prédit (q10, q50, q90) pour les ``horizon_weeks`` prochaines semaines.

    ``history_df`` doit contenir l'historique d'un seul SKU avec les colonnes
    standard du dataset (``stock_sku``, ``year``, ``week``, ``week_start``,
    ``usage``), trié par chronologie.

    Le modèle doit exposer ``predict_quantiles(X) -> DataFrame[q10, q50, q90]``.
    """
    if history_df["stock_sku"].nunique() != 1:
        raise ValueError("forecast_horizon attend un historique mono-SKU")
    if len(history_df) == 0:
        raise ValueError("Historique vide")

    sku = history_df["stock_sku"].iloc[0]
    sorted_hist = history_df.sort_values(["year", "week"]).reset_index(drop=True)
    usage_history = sorted_hist["usage"].astype("float64").tolist()
    last_year = int(sorted_hist["year"].iloc[-1])
    last_week = int(sorted_hist["week"].iloc[-1])

    rows = []
    for _ in range(horizon_weeks):
        next_year, next_week = _next_iso_week(last_year, last_week)
        from utils.sumup_shared import week_start  # pylint: disable=import-outside-toplevel
        week_start_dt = week_start(next_year, next_week)

        lag_values = {k: usage_history[-k] if len(usage_history) >= k else 0.0 for k in lags}
        feats = _row_features(
            sku=sku,
            year=next_year,
            week=next_week,
            week_start_dt=week_start_dt,
            lag_values=lag_values,
            rolling_history=usage_history,
            lags=lags,
            windows=windows,
        )
        X = pd.DataFrame([feats])
        q_df = model.predict_quantiles(X).iloc[0]
        q10, q50, q90 = float(q_df["q10"]), float(q_df["q50"]), float(q_df["q90"])

        rows.append({
            "stock_sku": sku,
            "year": next_year,
            "week": next_week,
            "week_start": week_start_dt,
            "q10": q10,
            "q50": q50,
            "q90": q90,
        })
        # Pour le pas suivant, on injecte la médiane comme "réalisation" observée.
        usage_history.append(q50)
        last_year, last_week = next_year, next_week

    return pd.DataFrame(rows)


def _sample_from_quantiles(q10: float, q50: float, q90: float, rng: np.random.Generator) -> float:
    """Échantillonne une valeur à partir des 3 quantiles (interpolation linéaire par morceaux)."""
    u = rng.random()
    if u <= 0.1:
        # tail gauche : extrapolation linéaire à partir de q10..q50
        return max(0.0, q10 - (q50 - q10) * (0.1 - u) / 0.4)
    if u <= 0.5:
        return q10 + (q50 - q10) * (u - 0.1) / 0.4
    if u <= 0.9:
        return q50 + (q90 - q50) * (u - 0.5) / 0.4
    # tail droite
    return q90 + (q90 - q50) * (u - 0.9) / 0.4


def simulate_rupture(
    stock_initial: float,
    weekly_quantiles: pd.DataFrame,
    incoming_qty: float = 0.0,
    incoming_eta: Optional[date] = None,
    n_simulations: int = 1000,
    seed: int = 0,
) -> dict:
    """Simule ``n_simulations`` trajectoires de stock pour estimer la date de rupture.

    Args:
        stock_initial: stock physique disponible aujourd'hui.
        weekly_quantiles: DataFrame produit par ``forecast_horizon`` avec ``week_start``,
            ``q10``, ``q50``, ``q90``.
        incoming_qty: quantité commandée à recevoir.
        incoming_eta: date d'arrivée prévue de ``incoming_qty`` (None = ignoré).
        n_simulations: nombre de trajectoires Monte-Carlo.
        seed: graine de reproductibilité.

    Returns:
        dict avec :
            ``rupture_date_p10`` (pessimiste) / ``rupture_date_p50`` /
            ``rupture_date_p90`` (optimiste), au format ``date`` (None si pas de
            rupture observée dans l'horizon).
            ``prob_rupture`` : probabilité d'avoir une rupture dans l'horizon.
            ``trajectories``: array (n_simulations, n_weeks) des stocks simulés.
    """
    if len(weekly_quantiles) == 0:
        raise ValueError("weekly_quantiles est vide")

    rng = np.random.default_rng(seed)
    n_weeks = len(weekly_quantiles)
    week_starts = list(weekly_quantiles["week_start"])
    incoming_week_idx = -1
    if incoming_eta is not None and incoming_qty > 0:
        for i, ws in enumerate(week_starts):
            if _to_date(ws) >= incoming_eta:
                incoming_week_idx = i
                break

    trajectories = np.zeros((n_simulations, n_weeks))
    rupture_weeks = np.full(n_simulations, -1, dtype=int)

    q10_arr = weekly_quantiles["q10"].to_numpy()
    q50_arr = weekly_quantiles["q50"].to_numpy()
    q90_arr = weekly_quantiles["q90"].to_numpy()

    for s in range(n_simulations):
        stock = float(stock_initial)
        for t in range(n_weeks):
            if t == incoming_week_idx:
                stock += float(incoming_qty)
            consumption = _sample_from_quantiles(q10_arr[t], q50_arr[t], q90_arr[t], rng)
            stock -= consumption
            trajectories[s, t] = max(stock, 0.0)
            if stock <= 0 and rupture_weeks[s] == -1:
                rupture_weeks[s] = t

    has_rupture = rupture_weeks >= 0
    prob_rupture = float(np.mean(has_rupture))

    def _percentile_date(p: float) -> Optional[date]:
        if not has_rupture.any():
            return None
        # On veut le quantile sur les semaines d'occurrence ; les "pas de rupture" sont
        # pris en compte en mettant n_weeks comme valeur (plus tard que tout).
        weeks_filled = np.where(has_rupture, rupture_weeks, n_weeks).astype(float)
        idx = float(np.percentile(weeks_filled, p, method="lower"))
        if idx >= n_weeks:
            return None
        return _to_date(week_starts[int(idx)])

    return {
        "rupture_date_p10": _percentile_date(10),
        "rupture_date_p50": _percentile_date(50),
        "rupture_date_p90": _percentile_date(90),
        "prob_rupture": prob_rupture,
        "n_simulations": n_simulations,
        "trajectories": trajectories,
    }


def _to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return date.fromisoformat(str(value))


__all__ = ["forecast_horizon", "simulate_rupture"]
