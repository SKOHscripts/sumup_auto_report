#!/usr/bin/env python3
"""Orchestrateur de l'inférence ML pour le rapport hebdomadaire.

Fait le pont entre le pipeline de génération de rapport (`stocks.sumup_stocks`)
et les modules ML (`features`, `model`, `projection`).

Workflow :

  1. Charger l'historique persistant (`stocks/ml/dataset.py`).
  2. Entraîner ou recharger un modèle quantile (TBD : registry + reload).
     Pour le moment, on entraîne à la volée à chaque appel — peu coûteux car
     HistGradientBoosting est rapide sur quelques milliers de lignes.
  3. Pour chaque SKU listé dans les KPIs : prévoir 26 semaines en avant et
     simuler la date de rupture.
  4. Renvoyer un dict `{stock_sku: ml_proj}` exploitable par le rendu PDF.

Le bloc complet est *best-effort* : si l'historique est trop court ou si une
erreur se produit, on retourne un dict vide et l'appelant retombe sur la
projection linéaire existante.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from stocks.ml.dataset import filter_skus, load_weekly_usage
from stocks.ml.features import DEFAULT_LAGS, prepare_training_table
from stocks.ml.model import QuantileGradientBoostingForecaster
from stocks.ml.projection import forecast_horizon, simulate_rupture

log = logging.getLogger(__name__)

# Nombre minimum de lignes (semaines x SKU) pour entraîner. En-dessous, on
# laisse la baseline historique s'occuper de la prévision.
MIN_TRAINING_ROWS = 30
# Nombre minimum de semaines pour un SKU donné : il faut au moins
# `max(DEFAULT_LAGS) + 4` semaines pour que `prepare_training_table` produise
# au moins quelques lignes utilisables.
MIN_WEEKS_PER_SKU = max(DEFAULT_LAGS) + 4
DEFAULT_HORIZON_WEEKS = 26
DEFAULT_N_SIMULATIONS = 1000


def train_global_model(
    history_df: pd.DataFrame,
    max_iter: int = 200,
    random_state: int = 0,
    config=None,
) -> QuantileGradientBoostingForecaster | None:
    """Entraîne le modèle quantile global sur l'historique fourni.

    Si ``config`` (``MLConfig``) est fourni, ses ``quantiles`` et
    ``tuned_params`` sont utilisés à la place des défauts. Sinon le tuning
    persisté sur disque est lu, ou les défauts sont appliqués.

    Retourne None si l'historique est insuffisant.
    """
    if len(history_df) < MIN_TRAINING_ROWS:
        log.info(
            "ML : historique insuffisant (%d lignes < %d), pas d'entrainement.",
            len(history_df), MIN_TRAINING_ROWS,
        )
        return None
    try:
        X, y, _ = prepare_training_table(history_df)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("ML : echec preparation des features (%s)", exc)
        return None
    if len(X) < MIN_TRAINING_ROWS:
        log.info("ML : table d'entrainement trop courte apres warm-up (%d lignes)", len(X))
        return None

    if config is None:
        from stocks.ml.config import load_config  # pylint: disable=import-outside-toplevel

        config = load_config()
    params = {**config.tuned_params}
    params["max_iter"] = params.get("max_iter", max_iter)

    try:
        model = QuantileGradientBoostingForecaster(
            quantiles=config.quantiles,
            random_state=random_state,
            **params,
        ).fit(X, y)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("ML : echec entrainement (%s)", exc)
        return None
    log.info(
        "ML : modele entraine (n_samples=%d, n_skus=%d, hash=%s)",
        model.metadata.n_samples, model.metadata.n_skus, model.metadata.config_hash,
    )
    return model


def project_for_sku(  # pylint: disable=too-many-arguments
    model: QuantileGradientBoostingForecaster,
    history_df: pd.DataFrame,
    sku: str,
    stock_initial: float,
    incoming_qty: float = 0.0,
    incoming_eta=None,
    horizon_weeks: int = DEFAULT_HORIZON_WEEKS,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int = 0,
) -> dict | None:
    """Calcule la projection ML pour un SKU. Retourne None si données insuffisantes."""
    sku_hist = history_df[history_df["stock_sku"] == sku]
    if len(sku_hist) < MIN_WEEKS_PER_SKU:
        return None
    try:
        forecast = forecast_horizon(model, sku_hist, horizon_weeks=horizon_weeks)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("ML : forecast %s echoue (%s)", sku, exc)
        return None
    quantile_fractions = (model.quantiles[0], model.quantiles[-1])
    sim = simulate_rupture(
        stock_initial=stock_initial,
        weekly_quantiles=forecast,
        incoming_qty=incoming_qty,
        incoming_eta=incoming_eta,
        n_simulations=n_simulations,
        seed=seed,
        quantile_fractions=quantile_fractions,
    )
    return {
        "rupture_date_p10": sim["rupture_date_p10"].isoformat() if sim["rupture_date_p10"] else None,
        "rupture_date_p50": sim["rupture_date_p50"].isoformat() if sim["rupture_date_p50"] else None,
        "rupture_date_p90": sim["rupture_date_p90"].isoformat() if sim["rupture_date_p90"] else None,
        "prob_rupture": sim["prob_rupture"],
        "weekly_forecast": forecast.assign(
            week_start=forecast["week_start"].apply(lambda d: d.isoformat()),
        ).to_dict(orient="records"),
        "model_version": model.metadata.config_hash,
        "model_trained_at": model.metadata.trained_at,
    }


def attach_ml_projections(
    all_kpis: list[dict],
    history_path: Path | None = None,
    skus: Iterable[str] | None = None,
) -> list[dict]:
    """Calcule et attache les projections ML à la liste de KPIs.

    Effet de bord : ajoute la clé ``ml_projection`` à chaque KPI éligible.
    Best-effort : retourne les KPIs inchangés si l'historique est absent ou
    si l'entraînement échoue.

    Si un modèle archivé existe (cf. ``stocks.ml.registry.load_current``), il
    est réutilisé tel quel (pas de réentraînement). Sinon on entraîne un
    modèle à la volée sur tout l'historique.
    """
    history = load_weekly_usage(history_path) if history_path else load_weekly_usage()
    if len(history) == 0:
        log.info("ML : pas d'historique persistant, projection ML desactivee.")
        return all_kpis

    if skus:
        history = filter_skus(history, skus)

    model = _load_or_train_model(history)
    if model is None:
        return all_kpis

    n_attached = 0
    for kpi in all_kpis:
        sku = kpi["stock_sku"]
        proj = project_for_sku(
            model,
            history_df=history,
            sku=sku,
            stock_initial=float(kpi.get("available_stock", 0.0) or 0.0),
            incoming_qty=float(kpi.get("incoming_qty", 0.0) or 0.0),
            incoming_eta=_parse_eta(kpi.get("incoming_eta")),
        )
        if proj is not None:
            kpi["ml_projection"] = proj
            n_attached += 1
    log.info("ML : projections attachees a %d/%d SKU.", n_attached, len(all_kpis))
    return all_kpis


def _load_or_train_model(history_df: pd.DataFrame) -> QuantileGradientBoostingForecaster | None:
    """Charge le modèle archivé via le registry, sinon entraîne à la volée."""
    try:
        from stocks.ml.registry import load_current  # pylint: disable=import-outside-toplevel
    except ImportError:
        return train_global_model(history_df)
    cached = load_current()
    if cached is not None:
        log.info("ML : modele charge depuis le registry (%s)", cached.metadata.config_hash)
        return cached
    return train_global_model(history_df)


def _parse_eta(value):
    if not value:
        return None
    from datetime import date  # pylint: disable=import-outside-toplevel

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


__all__ = [
    "DEFAULT_HORIZON_WEEKS",
    "DEFAULT_N_SIMULATIONS",
    "MIN_TRAINING_ROWS",
    "MIN_WEEKS_PER_SKU",
    "attach_ml_projections",
    "project_for_sku",
    "train_global_model",
]
