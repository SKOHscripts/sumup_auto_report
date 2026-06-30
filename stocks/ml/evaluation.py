#!/usr/bin/env python3
"""Évaluation des modèles de prévision : walk-forward backtest + métriques.

Métriques principales :
  - MAPE (Mean Absolute Percentage Error) sur le quantile médian (q50)
  - MAE (Mean Absolute Error) — robuste quand la cible est parfois nulle
  - Pinball loss par quantile (q10, q50, q90) — métrique propre pour modèles quantile
  - Coverage : proportion d'observations tombant dans [q10, q90] (cible ≈ 80 %)

Walk-forward : on coupe l'historique en K plis chronologiques, on entraîne
sur le passé, on prédit la fenêtre suivante, on accumule les métriques. Pas
de fuite de futur.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from stocks.ml.features import prepare_training_table
from stocks.ml.model import DEFAULT_QUANTILES, QuantileGradientBoostingForecaster

log = logging.getLogger(__name__)

# Seuils de qualité — promotion du modèle si métriques < seuil.
DEFAULT_MAPE_THRESHOLD = 0.45      # 45 % d'erreur relative médiane max (repli sans baseline)
DEFAULT_COVERAGE_TARGET = 0.80     # 80 % des observations dans [q_low, q_high]
DEFAULT_COVERAGE_TOLERANCE = 0.15  # tolérance ±15 pts
# Marge du critère relatif : sur une demande faible/erratique, le seuil MAPE
# absolu est inatteignable (la baseline elle-même ~70 %). On promeut donc le
# modèle s'il fait au moins aussi bien que la baseline, à cette marge près.
DEFAULT_RELATIVE_MAPE_MARGIN = 0.10


@dataclass
class EvaluationMetrics:
    """Résumé des métriques calculées sur un fold ou agrégées."""

    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    mean_bias: float = 0.0
    pinball_low: float = 0.0
    pinball_med: float = 0.0
    pinball_high: float = 0.0
    coverage_band: float = 0.0
    n_samples: int = 0
    n_folds: int = 0
    fold_metrics: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        """Renvoie les métriques sous forme de dict simple (sans fold_metrics)."""
        out = {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "mean_bias": self.mean_bias,
            "pinball_low": self.pinball_low,
            "pinball_med": self.pinball_med,
            "pinball_high": self.pinball_high,
            "coverage_band": self.coverage_band,
            "n_samples": self.n_samples,
            "n_folds": self.n_folds,
        }
        return out


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Biais moyen : erreur systématique (positif = surestimation)."""
    return float(np.mean(y_pred - y_true))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE robuste : ajoute ``eps`` au dénominateur pour gérer y_true ≈ 0."""
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Perte pinball (asymétrique) pour le quantile ``q``."""
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Proportion d'observations couvertes par l'intervalle [y_low, y_high]."""
    inside = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(inside))


def _fold_indices(meta: pd.DataFrame, n_folds: int, min_train_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construit les indices (train, test) par fold temporel.

    Stratégie : on classe par (year, week), puis on découpe la fin de
    l'historique en ``n_folds`` segments contigus. Chaque pli prédit son
    segment à partir de tout ce qui le précède (train croissant).
    """
    order = meta.sort_values(["year", "week"]).index.to_numpy()
    if len(order) < min_train_size + n_folds:
        return []
    test_size = max(1, (len(order) - min_train_size) // n_folds)
    folds = []
    for i in range(n_folds):
        train_end = min_train_size + i * test_size
        test_end = min(train_end + test_size, len(order))
        if train_end >= len(order):
            break
        train_idx = order[:train_end]
        test_idx = order[train_end:test_end]
        if len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def walk_forward_backtest(
    history_df: pd.DataFrame,
    n_folds: int = 5,
    min_train_size: int = 50,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    max_iter: int = 100,
    random_state: int = 0,
    target_transform: str | None = None,
    model_params: dict | None = None,
) -> EvaluationMetrics:
    """Évalue ``QuantileGradientBoostingForecaster`` en walk-forward.

    ``model_params`` permet d'évaluer **exactement** le modèle qui sera déployé
    (hyperparamètres tunés : max_depth, l2, etc.). Sans lui, l'évaluation
    utiliserait des valeurs par défaut différentes du modèle final, ce qui
    fausse la décision de promotion.
    """
    params = dict(model_params or {})
    params.setdefault("max_iter", max_iter)
    X, y, meta = prepare_training_table(history_df)
    folds = _fold_indices(meta, n_folds=n_folds, min_train_size=min_train_size)
    if not folds:
        log.info(
            "Backtest : pas assez de donnees (n=%d, min_train=%d, n_folds=%d)",
            len(X), min_train_size, n_folds,
        )
        return EvaluationMetrics()

    quantile_list = sorted(quantiles)
    if len(quantile_list) != 3:
        raise ValueError("walk_forward_backtest attend exactement 3 quantiles (low, med, high)")
    q_low_frac, q_med_frac, q_high_frac = quantile_list
    fold_results: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        model = QuantileGradientBoostingForecaster(
            quantiles=tuple(quantile_list),
            random_state=random_state,
            target_transform=target_transform,
            **params,
        ).fit(X.loc[train_idx], y.loc[train_idx])
        preds = model.predict_quantiles(X.loc[test_idx])
        y_test = y.loc[test_idx].to_numpy()
        q_low_arr = preds["q_low"].to_numpy()
        q_med_arr = preds["q_med"].to_numpy()
        q_high_arr = preds["q_high"].to_numpy()
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "mae": mae(y_test, q_med_arr),
            "rmse": rmse(y_test, q_med_arr),
            "mape": mape(y_test, q_med_arr),
            "mean_bias": mean_bias(y_test, q_med_arr),
            "pinball_low": pinball_loss(y_test, q_low_arr, q_low_frac),
            "pinball_med": pinball_loss(y_test, q_med_arr, q_med_frac),
            "pinball_high": pinball_loss(y_test, q_high_arr, q_high_frac),
            "coverage_band": coverage(y_test, q_low_arr, q_high_arr),
        })

    n = len(fold_results)
    return EvaluationMetrics(
        mae=float(np.mean([r["mae"] for r in fold_results])),
        rmse=float(np.mean([r["rmse"] for r in fold_results])),
        mape=float(np.mean([r["mape"] for r in fold_results])),
        mean_bias=float(np.mean([r["mean_bias"] for r in fold_results])),
        pinball_low=float(np.mean([r["pinball_low"] for r in fold_results])),
        pinball_med=float(np.mean([r["pinball_med"] for r in fold_results])),
        pinball_high=float(np.mean([r["pinball_high"] for r in fold_results])),
        coverage_band=float(np.mean([r["coverage_band"] for r in fold_results])),
        n_samples=int(sum(r["n_test"] for r in fold_results)),
        n_folds=n,
        fold_metrics=fold_results,
    )


def baseline_avg_rolling4(history_df: pd.DataFrame) -> float:
    """MAPE de la baseline historique (moyenne mobile 4 sem.) en walk-forward simple."""
    df = history_df.sort_values(["stock_sku", "year", "week"]).copy()
    df["pred"] = df.groupby("stock_sku")["usage"].shift(1).rolling(window=4, min_periods=1).mean()
    df = df.dropna(subset=["pred"])
    if len(df) == 0:
        return float("nan")
    return mape(df["usage"].to_numpy(), df["pred"].to_numpy())


def is_model_promotable(
    metrics: EvaluationMetrics,
    baseline_mape: float | None = None,
    mape_threshold: float = DEFAULT_MAPE_THRESHOLD,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    relative_mape_margin: float = DEFAULT_RELATIVE_MAPE_MARGIN,
) -> tuple[bool, list[str]]:
    """Décide si le modèle peut être promu en remplacement du précédent.

    Critères :
      1. Précision — critère **relatif** à la baseline si elle est fournie :
         la MAPE du modèle doit rester ``<= baseline * (1 + marge)`` (le modèle
         fait au moins aussi bien que la moyenne mobile, à la marge près). En
         l'absence de baseline, on retombe sur le seuil **absolu**
         ``mape_threshold``.
      2. Couverture [q_low, q_high] dans ``[target ± tolerance]``.

    Le critère relatif est adapté à une demande faible et erratique, où la MAPE
    absolue est structurellement élevée (la baseline elle-même ~70 %) : un seuil
    absolu serait inatteignable quel que soit le modèle.

    Retourne ``(promotable, [raisons d'echec])``.
    """
    reasons: list[str] = []
    if metrics.n_folds == 0:
        reasons.append("aucun fold valide")
        return False, reasons

    has_baseline = baseline_mape is not None and not np.isnan(baseline_mape)
    if has_baseline:
        allowed = baseline_mape * (1.0 + relative_mape_margin)
        if metrics.mape > allowed:
            reasons.append(
                f"MAPE ({metrics.mape:.2%}) au-dessus de la baseline +marge "
                f"({baseline_mape:.2%} +{relative_mape_margin:.0%} = {allowed:.2%})"
            )
    elif metrics.mape > mape_threshold:
        reasons.append(f"MAPE trop eleve ({metrics.mape:.2%} > {mape_threshold:.0%})")

    if abs(metrics.coverage_band - coverage_target) > coverage_tolerance:
        reasons.append(
            f"Coverage hors cible ({metrics.coverage_band:.0%} vs "
            f"{coverage_target:.0%}±{coverage_tolerance:.0%})"
        )
    return (len(reasons) == 0), reasons


__all__ = [
    "DEFAULT_COVERAGE_TARGET",
    "DEFAULT_COVERAGE_TOLERANCE",
    "DEFAULT_MAPE_THRESHOLD",
    "EvaluationMetrics",
    "baseline_avg_rolling4",
    "coverage",
    "is_model_promotable",
    "mae",
    "mean_bias",
    "mape",
    "pinball_loss",
    "rmse",
    "walk_forward_backtest",
]
