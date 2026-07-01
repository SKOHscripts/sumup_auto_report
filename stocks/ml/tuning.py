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
import math

import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
# Active l'API halving (import à effet de bord requis avant HalvingRandomSearchCV).
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  pylint: disable=unused-import
from sklearn.model_selection import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)

from stocks.ml.config import DEFAULT_HGB_PARAMS, MLConfig, load_config, save_config
from stocks.ml.evaluation import walk_forward_backtest
from stocks.ml.features import prepare_training_table

log = logging.getLogger(__name__)

PARAM_GRID = {
    # On couvre de "rapide et peu d'arbres" à "lent et beaucoup d'arbres"
    "max_iter": [100, 200, 300, 500, 800, 1000],
    # Profondeur de très simple (2) à illimitée (None)
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    # De très prudent à assez agressif
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    # Feuilles de petite taille (modèle plus flexible) à très grosses (très lissant)
    "min_samples_leaf": [3, 5, 10, 20, 40, 80],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    "max_leaf_nodes": [15, 31, 63, None],  # None = pas de limite
}


def _pinball_score(q: float):
    """Construit un scorer sklearn pour la pinball loss (négatif car sklearn maximise).

    Le score est calculé dans l'espace d'entraînement du modèle. Avec une cible
    log1p, c'est donc l'espace log : la pinball y est **équilibrée entre SKU**
    (les gros volumes ne dominent pas), ce qui aligne le tuning sur l'erreur
    relative (MAPE) plutôt que sur l'erreur absolue dominée par le café.
    """
    def _score(estimator, X, y):
        preds = estimator.predict(X)
        diff = y - preds

        return -float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

    return _score


def _grid_size(grid: dict) -> int:
    """Calcule le nombre total de combinaisons d'une grille."""

    return math.prod(len(v) for v in grid.values())


def _backtest_mape(history_df: pd.DataFrame, cfg: MLConfig, params: dict) -> float:
    """MAPE walk-forward d'un jeu d'hyperparamètres (inf si pas assez de données)."""
    metrics = walk_forward_backtest(
        history_df,
        n_folds=5,
        quantiles=cfg.quantiles,
        target_transform=cfg.target_transform,
        model_params=params,
    )
    return metrics.mape if metrics.n_folds else float("inf")


def tune_hyperparameters(  # pylint: disable=too-many-arguments,too-many-locals
    history_df: pd.DataFrame,
    n_iter: int = None,
    n_splits: int = 4,
    target_quantile: float = 0.5,
    sku_col: str = "stock_sku",
    random_state: int = 0,
    grid: Iterable[dict] | dict | None = None,
    target_transform: str | None = None,
    search: str = "halving",
    n_jobs: int = -1,
    factor: int = 3,
) -> tuple[dict, float]:
    """Cherche les meilleurs hyperparamètres HGB sur le quantile cible.

    ``search`` :
      - ``"halving"`` (défaut) : successive halving avec ``max_iter`` comme
        ressource. Beaucoup de combinaisons sont testées à petit budget (peu
        d'arbres), et seules les meilleures sont ré-évaluées à plein budget →
        ~10× plus rapide, donc bien plus d'échantillonnage à temps égal.
      - ``"random"`` : RandomizedSearchCV classique (repli).

    ``n_jobs`` répartit les fits sur les cœurs (``-1`` = tous). Pour distribuer
    sur plusieurs machines, encapsuler l'appel dans un backend joblib
    (``with joblib.parallel_backend("dask"): ...``) — aucun autre changement.

    Retourne ``(best_params, best_score)`` où ``best_score`` est la pinball
    loss (positive) — plus c'est petit, mieux c'est.
    """
    grid = dict(grid or PARAM_GRID)

    X, y, _ = prepare_training_table(history_df)
    if target_transform == "log1p":
        y = np.log1p(y)
    if X[sku_col].dtype.name != "category":
        X = X.copy()
        X[sku_col] = X[sku_col].astype("category")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    base = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=target_quantile,
        random_state=random_state,
        categorical_features=[sku_col],
    )
    scorer = _pinball_score(target_quantile)

    if search in ("halving", "halving_grid"):
        # max_iter devient la « ressource » : on le retire de la grille.
        max_iter_vals = grid.pop("max_iter", [1000])
        max_resources = int(max(max_iter_vals))
        min_resources = max(20, max_resources // (factor ** 3))
        total = _grid_size(grid)
        common = {
            "estimator": base, "resource": "max_iter", "factor": factor,
            "max_resources": max_resources, "min_resources": min_resources,
            "scoring": scorer, "cv": splitter, "random_state": random_state,
            "n_jobs": n_jobs, "refit": False,
        }
        # Exhaustif si demandé explicitement, ou si n_candidates couvre toute la grille.
        exhaustive = search == "halving_grid" or (n_iter is not None and n_iter >= total)
        if exhaustive:
            log.info("Halving GRID exhaustif : %d combinaisons (hors max_iter)", total)
            estimator = HalvingGridSearchCV(param_grid=grid, **common).fit(X, y)
        else:
            n_cand = min(n_iter, total) if n_iter else "exhaust"
            estimator = HalvingRandomSearchCV(
                param_distributions=grid, n_candidates=n_cand, **common,
            ).fit(X, y)
            log.info(
                "Halving aleatoire : %s candidats initiaux (grille=%d), ressource max_iter<=%d",
                getattr(estimator, "n_candidates_", ["?"])[0], total, max_resources,
            )
        best_params = dict(estimator.best_params_)
        # La ressource (max_iter) n'est pas dans best_params_ : on l'ajoute.
        best_params["max_iter"] = int(getattr(estimator, "max_resources_", max_resources))
    else:
        total = _grid_size(grid)
        effective_n_iter = min(n_iter, total) if n_iter else total
        estimator = RandomizedSearchCV(
            estimator=base,
            param_distributions=grid,
            n_iter=effective_n_iter,
            scoring=scorer,
            cv=splitter,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=False,
        ).fit(X, y)
        best_params = dict(estimator.best_params_)

    best_score = -float(estimator.best_score_)
    log.info("Tuning : meilleurs params = %s (pinball=%.4f)", best_params, best_score)

    return best_params, best_score


def tune_and_save(history_df: pd.DataFrame,
                  n_candidates: int = 300,
                  n_jobs: int = -1,
                  exhaustive: bool = False,
                  config_path=None) -> MLConfig:
    """Tune (successive halving) puis persiste la config si elle s'améliore.

    ``n_candidates`` combinaisons sont échantillonnées puis filtrées par
    halving (peu coûteux). Si ``exhaustive=True``, TOUTE la grille est balayée
    (HalvingGridSearchCV) au lieu d'un échantillon. ``n_jobs`` répartit les fits
    sur les cœurs. Retourne la ``MLConfig`` (mise à jour seulement si le
    backtest s'améliore).
    """
    cfg = load_config(config_path)
    log.info(
        "Tuning halving : %s, n_jobs=%s, transform=%s...",
        "GRILLE EXHAUSTIVE" if exhaustive else f"{n_candidates} candidats",
        n_jobs, cfg.target_transform,
    )
    best_params, best_score = tune_hyperparameters(
        history_df,
        n_iter=n_candidates,
        grid=PARAM_GRID,
        target_transform=cfg.target_transform,
        search="halving_grid" if exhaustive else "halving",
        n_jobs=n_jobs,
    )
    log.info("Fin tuning : pinball=%.4f, params=%s", best_score, best_params)

    # Garde-fou : on n'adopte les nouveaux hyperparamètres que s'ils améliorent
    # réellement le backtest (MAPE) face aux params actuels. La CV pinball du
    # tuner ne capture pas toujours la généralisation ; sans ce filet, --tune
    # pourrait dégrader une config déjà bonne.
    candidate = {**DEFAULT_HGB_PARAMS, **best_params}
    current_mape = _backtest_mape(history_df, cfg, cfg.tuned_params)
    candidate_mape = _backtest_mape(history_df, cfg, candidate)
    log.info(
        "Validation tuning : MAPE actuelle=%.2f%% vs candidate=%.2f%%",
        current_mape * 100, candidate_mape * 100,
    )

    if candidate_mape >= current_mape:
        log.warning(
            "Tuning ignore : les hyperparametres actuels sont au moins aussi bons "
            "(MAPE %.2f%% <= %.2f%%). Config inchangee.",
            current_mape * 100, candidate_mape * 100,
        )
        return cfg

    cfg.tuned_params = candidate
    cfg.tuned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg.tuning_score = best_score
    save_config(cfg, config_path)
    log.info(
        "Config ML mise a jour dans %s (MAPE %.2f%% -> %.2f%%, tuned_at=%s)",
        config_path, current_mape * 100, candidate_mape * 100, cfg.tuned_at,
    )

    return cfg


__all__ = ["PARAM_GRID", "tune_and_save", "tune_hyperparameters"]
