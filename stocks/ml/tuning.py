#!/usr/bin/env python3
"""Recherche d'hyperparamètres pour le modèle quantile multi-SKU.

Usage typique via la CLI :

    python -m stocks.ml.train --tune

Le tuning :
  1. construit la table d'entrainement (features + lags),
  2. tire ``n_iter`` combinaisons d'hyperparamètres dans une grille,
  3. les évalue en TimeSeriesSplit sur le quantile médian (loss=quantile, q=0.5),
     en utilisant la **pinball loss** comme score (cohérent avec l'inférence
     finale qui produit aussi les autres quantiles).
  4. persiste le meilleur jeu dans ``stocks/models/config.json`` via
     ``stocks.ml.config``.

On tune **uniquement le modèle médian** : les hyperparamètres sont ensuite
réutilisés tels quels pour entraîner les modèles q_low et q_high. Cela évite
d'avoir 3× plus de combinaisons à tester (acceptable car HGB se comporte
similairement aux 3 quantiles dans la plupart des cas).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from stocks.ml.config import DEFAULT_HGB_PARAMS, MLConfig, load_config, save_config
from stocks.ml.features import prepare_training_table

log = logging.getLogger(__name__)

# Grille raisonnable : ~108 combinaisons. RandomizedSearchCV en explore n_iter.
PARAM_GRID = {
    "max_iter": [100, 200, 300, 500],
    "max_depth": [3, 4, 6, None],
    "learning_rate": [0.02, 0.05, 0.1],
    "min_samples_leaf": [3, 5, 10, 20],
}


def _pinball_score(q: float):
    """Construit un scorer sklearn pour la pinball loss (négatif car sklearn maximise)."""
    def _score(estimator, X, y):
        preds = estimator.predict(X)
        diff = y - preds
        return -float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return _score


def tune_hyperparameters(
    history_df: pd.DataFrame,
    n_iter: int = 20,
    n_splits: int = 4,
    target_quantile: float = 0.5,
    sku_col: str = "stock_sku",
    random_state: int = 0,
    grid: Iterable[dict] | dict | None = None,
) -> tuple[dict, float]:
    """Cherche les meilleurs hyperparamètres HGB sur le quantile cible.

    Retourne ``(best_params, best_score)`` où ``best_score`` est la pinball
    loss (positive) — plus c'est petit, mieux c'est.
    """
    X, y, _ = prepare_training_table(history_df)
    if X[sku_col].dtype.name != "category":
        X = X.copy()
        X[sku_col] = X[sku_col].astype("category")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    grid = grid or PARAM_GRID

    base = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=target_quantile,
        random_state=random_state,
        categorical_features=[sku_col],
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=grid,
        n_iter=n_iter,
        scoring=_pinball_score(target_quantile),
        cv=splitter,
        random_state=random_state,
        n_jobs=1,
        refit=False,
    )
    search.fit(X, y)
    best_params = dict(search.best_params_)
    best_score = -float(search.best_score_)
    log.info(
        "Tuning : meilleurs params = %s (pinball=%.4f)",
        best_params, best_score,
    )
    return best_params, best_score


def tune_and_save(
    history_df: pd.DataFrame,
    n_iter: int = 20,
    config_path=None,
) -> MLConfig:
    """Tune puis persiste la config. Retourne la ``MLConfig`` mise à jour."""
    cfg = load_config(config_path)
    best_params, best_score = tune_hyperparameters(history_df, n_iter=n_iter)
    cfg.tuned_params = {**DEFAULT_HGB_PARAMS, **best_params}
    cfg.tuned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg.tuning_score = best_score
    save_config(cfg, config_path)
    return cfg


__all__ = ["PARAM_GRID", "tune_and_save", "tune_hyperparameters"]
