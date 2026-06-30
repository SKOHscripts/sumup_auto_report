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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

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


def tune_hyperparameters(
    history_df: pd.DataFrame,
    n_iter: int = None,
    n_splits: int = 4,
    target_quantile: float = 0.5,
    sku_col: str = "stock_sku",
    random_state: int = 0,
    grid: Iterable[dict] | dict | None = None,
    target_transform: str | None = None,
) -> tuple[dict, float]:
    """Cherche les meilleurs hyperparamètres HGB sur le quantile cible.

    Retourne ``(best_params, best_score)`` où ``best_score`` est la pinball
    loss (positive, échelle d'origine) — plus c'est petit, mieux c'est.
    """
    total_combinations = _grid_size(grid)

    # Si n_iter non spécifié ou supérieur au total → on teste tout
    effective_n_iter = min(n_iter, total_combinations) if n_iter else total_combinations

    log.info(
        "RandomizedSearchCV: %d/%d combinaisons testées (%s)",
        effective_n_iter,
        total_combinations,
        "EXHAUSTIF" if effective_n_iter == total_combinations else "échantillonnage",
    )

    X, y, _ = prepare_training_table(history_df)
    if target_transform == "log1p":
        y = np.log1p(y)

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
        n_iter=effective_n_iter,
        scoring=_pinball_score(target_quantile),
        cv=splitter,
        random_state=random_state,
        n_jobs=-1,  # all processors
        verbose=2,
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


def build_fine_grid(best_params: dict) -> dict:
    """Grille fine locale autour de best_params.

    Paramètres attendus dans best_params:
      - max_iter (int)
      - max_depth (int | None)
      - learning_rate (float)
      - min_samples_leaf (int)
      - l2_regularization (float)  <- nouveau
      - max_leaf_nodes (int | None) <- nouveau
    """
    max_iter = int(best_params.get("max_iter", 200))
    max_depth = best_params.get("max_depth", 4)
    lr = float(best_params.get("learning_rate", 0.05))
    min_leaf = int(best_params.get("min_samples_leaf", 20))
    l2 = float(best_params.get("l2_regularization", 0.0))
    max_leaf = best_params.get("max_leaf_nodes", 31)

    # --- max_iter : petits pas autour de la valeur ---

    if max_iter <= 300:
        step = 25
    elif max_iter <= 600:
        step = 50
    else:
        step = 100
    max_iter_vals = sorted({
        max(50, max_iter - 2 * step),
        max(50, max_iter - step),
        max_iter,
        max_iter + step,
        max_iter + 2 * step,
    })

    # --- max_depth : voisinage immédiat ---

    if max_depth is None:
        max_depth_vals = [3, 4, 5, None]
    else:
        max_depth_vals = sorted({
            max(2, max_depth - 1),
            max_depth,
            max_depth + 1,
        })
        max_depth_vals.append(None)

    # --- learning_rate : +- 20% et +- 40% ---
    lr_step = lr * 0.2
    lr_vals = sorted({
        round(max(lr - 2 * lr_step, 0.005), 3),
        round(max(lr - lr_step, 0.005), 3),
        round(lr, 3),
        round(min(lr + lr_step, 0.2), 3),
        round(min(lr + 2 * lr_step, 0.2), 3),
    })

    # --- min_samples_leaf : petits incréments entiers ---

    if min_leaf <= 10:
        leaf_candidates = {
            max(2, min_leaf - 2),
            max(2, min_leaf - 1),
            min_leaf,
            min_leaf + 1,
            min_leaf + 2,
        }
    else:
        leaf_step = max(2, min_leaf // 5)
        leaf_candidates = {
            max(2, min_leaf - 2 * leaf_step),
            max(2, min_leaf - leaf_step),
            min_leaf,
            min_leaf + leaf_step,
            min_leaf + 2 * leaf_step,
        }
    min_leaf_vals = sorted(leaf_candidates)

    # --- l2_regularization : voisinage multiplicatif (échelle log) ---

    if l2 == 0.0:
        # Si coarse a choisi 0, on teste autour de 0 et les petites valeurs
        l2_vals = [0.0, 0.05, 0.1, 0.2, 0.5]
    else:
        l2_vals = sorted({
            0.0,                             # toujours tester 0 comme référence
            round(max(l2 * 0.5, 0.01), 3),
            round(l2, 3),
            round(min(l2 * 2.0, 10.0), 3),
            round(min(l2 * 4.0, 10.0), 3),
        })

    # --- max_leaf_nodes : voisinage entier ---

    if max_leaf is None:
        max_leaf_vals = [31, 63, 127, None]
    else:
        max_leaf_vals = sorted({
            max(7, max_leaf // 2),
            max(7, max_leaf - 8),
            max_leaf,
            max_leaf + 8,
            max_leaf * 2,
        })
        max_leaf_vals.append(None)

    return {
        "max_iter": max_iter_vals,
        "max_depth": max_depth_vals,
        "learning_rate": lr_vals,
        "min_samples_leaf": min_leaf_vals,
        "l2_regularization": l2_vals,
        "max_leaf_nodes": max_leaf_vals,
    }


def tune_and_save(history_df: pd.DataFrame,
                  n_iter_coarse: int = 200,
                  n_iter_fine: int = None,
                  config_path=None) -> MLConfig:
    """Tune puis persiste la config. Retourne la ``MLConfig`` mise à jour."""
    cfg = load_config(config_path)
    transform = cfg.target_transform

    # 1) Coarse search avec la grosse grille
    log.info(
        "Tuning coarse: n_iter=%s sur grille grossiere (RandomizedSearchCV, transform=%s)...",
        n_iter_coarse, transform,
    )
    coarse_best, coarse_score = tune_hyperparameters(
        history_df,
        n_iter=n_iter_coarse,
        grid=PARAM_GRID,   # la grosse grille
        target_transform=transform,
    )
    log.info(
        "Fin coarse: pinball=%.4f, params=%s",
        coarse_score, coarse_best,
    )

    # 2) Fine search autour de coarse_best
    fine_grid = build_fine_grid(coarse_best)
    log.info(
        "Tuning fine: n_iter=%s autour du best coarse (grid taille=%d)...",
        n_iter_fine,
        _grid_size(fine_grid)
    )
    fine_best, fine_score = tune_hyperparameters(
        history_df,
        n_iter=n_iter_fine,
        grid=fine_grid,
        target_transform=transform,
    )
    log.info(
        "Fin fine: pinball=%.4f, params=%s",
        fine_score, fine_best,
    )

    # On garde le meilleur des deux (score = pinball loss, donc plus petit = mieux)

    if fine_score < coarse_score:
        phase = "fine"
        best_params, best_score = fine_best, fine_score
    else:
        phase = "coarse"
        best_params, best_score = coarse_best, coarse_score

    log.info(
        "Tuning termine (phase retenue=%s): pinball=%.4f, params=%s",
        phase, best_score, best_params,
    )

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
