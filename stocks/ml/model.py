#!/usr/bin/env python3
"""Modèles de prévision de consommation hebdomadaire.

Pour le moment, un seul modèle de référence (baseline) :

    RidgeForecaster
        Régression linéaire régularisée multi-SKU. Encode `stock_sku` en
        one-hot puis pipe à Ridge avec standardisation. Prévision ponctuelle.

Conventions :
    - X / y / meta sont produits par ``stocks.ml.features.prepare_training_table``.
    - Les modèles exposent ``fit(X, y)`` et ``predict(X) -> np.ndarray``.
    - ``save(path) / load(path)`` sérialisent via joblib + un dict de métadonnées
      (date, version sklearn, hash de la config) à côté du modèle.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelMetadata:
    """Métadonnées sérialisées à côté du modèle pour le suivi des versions."""

    trained_at: str = ""
    sklearn_version: str = ""
    n_samples: int = 0
    n_skus: int = 0
    n_features: int = 0
    config_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:12]


class RidgeForecaster:
    """Baseline : Ridge multi-SKU avec one-hot du stock_sku.

    Usage typique ::

        X, y, meta = prepare_training_table(history_df)
        model = RidgeForecaster(alpha=1.0).fit(X, y)
        y_pred = model.predict(X_future)
    """

    def __init__(self, alpha: float = 1.0, sku_col: str = "stock_sku"):
        self.alpha = alpha
        self.sku_col = sku_col
        self.pipeline_: Pipeline | None = None
        self.metadata = ModelMetadata()

    def _build_pipeline(self, feature_cols: list[str]) -> Pipeline:
        numeric_cols = [c for c in feature_cols if c != self.sku_col]
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        pre = ColumnTransformer(
            transformers=[
                ("sku", ohe, [self.sku_col]),
                ("num", StandardScaler(), numeric_cols),
            ],
            remainder="drop",
        )
        return Pipeline([("pre", pre), ("ridge", Ridge(alpha=self.alpha))])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        """Entraîne le modèle sur (X, y). Retourne self."""
        if self.sku_col not in X.columns:
            raise ValueError(f"Colonne `{self.sku_col}` absente de X")
        self.pipeline_ = self._build_pipeline(list(X.columns))
        self.pipeline_.fit(X, y.astype("float64"))
        self.metadata = ModelMetadata(
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sklearn_version=sklearn.__version__,
            n_samples=int(len(X)),
            n_skus=int(X[self.sku_col].nunique()),
            n_features=int(X.shape[1]),
            config_hash=_config_hash({"alpha": self.alpha, "model": "ridge"}),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit la consommation hebdomadaire pour chaque ligne de X."""
        if self.pipeline_ is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné. Appelez `.fit` d'abord.")
        preds = self.pipeline_.predict(X)
        # Une consommation ne peut pas être négative.
        return np.clip(preds, a_min=0.0, a_max=None)

    def save(self, path: Path | str) -> Path:
        """Sérialise le pipeline + un fichier `<path>.meta.json` à côté."""
        if self.pipeline_ is None:
            raise RuntimeError("Rien à sauvegarder : modèle non entraîné.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline_, target)
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "RidgeForecaster":
        """Charge un modèle entraîné précédemment via `.save`."""
        target = Path(path)
        instance = cls()
        instance.pipeline_ = joblib.load(target)
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                instance.metadata = ModelMetadata(**json.load(f))
        return instance


__all__ = ["ModelMetadata", "RidgeForecaster"]
