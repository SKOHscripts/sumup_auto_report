#!/usr/bin/env python3
"""Configuration persistante du pipeline ML.

Stocke dans ``stocks/models/config.json`` les hyperparamètres tunés et les
seuils de promotion. Permet à ``stocks.ml.train`` de réutiliser les meilleurs
réglages d'une exécution sur l'autre, sans repasser par ``--tune`` à chaque
fois.

Schéma JSON minimal ::

    {
        "quantiles": [0.05, 0.5, 0.95],
        "mape_threshold": 0.45,
        "coverage_target": 0.80,
        "coverage_tolerance": 0.10,
        "tuned_params": {
            "max_iter": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_samples_leaf": 5
        },
        "tuned_at": "2026-05-03T10:00:00+00:00",
        "tuning_score": null
    }

Les valeurs absentes sont remplacées par les défauts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "models"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
DEFAULT_HGB_PARAMS = {
    "max_iter": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
}


@dataclass
class MLConfig:
    """Réglages persistants du pipeline ML."""

    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES
    mape_threshold: float = 0.45
    coverage_target: float = 0.80
    coverage_tolerance: float = 0.15
    relative_mape_margin: float = 0.10
    tuned_params: dict = field(default_factory=lambda: dict(DEFAULT_HGB_PARAMS))
    tuned_at: Optional[str] = None
    tuning_score: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.quantiles, list):
            self.quantiles = tuple(self.quantiles)  # type: ignore[assignment]
        if len(self.quantiles) != 3:
            raise ValueError(f"quantiles doit avoir exactement 3 valeurs, recu : {self.quantiles}")
        # On garantit l'ordre croissant et que la mediane vaut 0.5.
        sq = tuple(sorted(self.quantiles))
        if abs(sq[1] - 0.5) > 1e-6:
            raise ValueError(f"Le quantile median doit etre 0.5, recu : {sq[1]}")
        self.quantiles = sq  # type: ignore[assignment]

    @property
    def quantile_fractions(self) -> tuple[float, float]:
        """Renvoie ``(q_low_frac, q_high_frac)`` pour le sampler Monte-Carlo."""
        return (self.quantiles[0], self.quantiles[2])

    def as_dict(self) -> dict:
        """Sérialise la config en dict prêt pour ``json.dumps`` (tuple → list)."""
        out = asdict(self)
        out["quantiles"] = list(self.quantiles)
        return out


def load_config(path: Path | None = None) -> MLConfig:
    """Charge la config depuis ``path`` (ou ``CONFIG_PATH`` par défaut).

    Retourne une ``MLConfig`` aux valeurs par défaut si le fichier n'existe pas
    ou est invalide.
    """
    target = Path(path) if path else CONFIG_PATH
    if not target.exists():
        return MLConfig()
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Config ML illisible (%s), utilisation des defauts : %s", target, exc)
        return MLConfig()

    cfg = MLConfig()
    cfg.quantiles = tuple(sorted(data.get("quantiles", list(DEFAULT_QUANTILES))))  # type: ignore[assignment]
    cfg.mape_threshold = float(data.get("mape_threshold", cfg.mape_threshold))
    cfg.coverage_target = float(data.get("coverage_target", cfg.coverage_target))
    cfg.coverage_tolerance = float(data.get("coverage_tolerance", cfg.coverage_tolerance))
    cfg.relative_mape_margin = float(data.get("relative_mape_margin", cfg.relative_mape_margin))
    tuned = data.get("tuned_params") or {}
    if isinstance(tuned, dict):
        cfg.tuned_params = {**DEFAULT_HGB_PARAMS, **tuned}
    cfg.tuned_at = data.get("tuned_at")
    score = data.get("tuning_score")
    cfg.tuning_score = float(score) if score is not None else None
    cfg.__post_init__()  # revalidation
    return cfg


def save_config(config: MLConfig, path: Path | None = None) -> Path:
    """Persiste la config sur disque (création du dossier si besoin)."""
    target = Path(path) if path else CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(config.as_dict(), f, indent=2, ensure_ascii=False, sort_keys=True)
    return target


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_HGB_PARAMS",
    "DEFAULT_QUANTILES",
    "MLConfig",
    "load_config",
    "save_config",
]
