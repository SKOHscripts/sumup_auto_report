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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
QUANTILE_LABELS = ("q_low", "q_med", "q_high")


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
    return hashlib.sha256(payload).hexdigest()[:12]


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


class QuantileGradientBoostingForecaster:
    """Modèle global multi-SKU à 3 quantiles (q10, q50, q90).

    Chaque quantile est entraîné indépendamment via
    ``HistGradientBoostingRegressor(loss="quantile", quantile=q)``. La sortie
    de ``predict_quantiles(X)`` est un DataFrame avec les colonnes ``q10``,
    ``q50``, ``q90`` qui permettent de construire un intervalle de confiance
    et de simuler des trajectoires (cf. ``stocks.ml.projection``).

    Le ``stock_sku`` est passé en feature catégorielle native (pas besoin de
    one-hot : HGB gère les catégories).
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        max_iter: int = 200,
        max_depth: int | None = 6,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 5,
        l2_regularization: float = 0.0,   # NOUVEAU
        max_leaf_nodes: int = 31,        # NOUVEAU
        random_state: int = 0,
        sku_col: str = "stock_sku",
    ):
        self.quantiles = tuple(quantiles)
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.sku_col = sku_col
        self.models_: dict[float, HistGradientBoostingRegressor] = {}
        self.feature_cols_: list[str] = []
        self.metadata = ModelMetadata()
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes

    def _make_estimator(self, q: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=getattr(self, "l2_regularization", 0.0),
            max_leaf_nodes=getattr(self, "max_leaf_nodes", 31),
            random_state=self.random_state,
            categorical_features=[self.sku_col],
        )

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        if self.sku_col in out.columns:
            out[self.sku_col] = out[self.sku_col].astype("category")

        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileGradientBoostingForecaster":
        """Entraîne un HGB par quantile."""

        if self.sku_col not in X.columns:
            raise ValueError(f"Colonne `{self.sku_col}` absente de X")
        prepared = self._prepare_features(X)
        self.feature_cols_ = list(prepared.columns)
        y_arr = y.astype("float64").to_numpy()

        for q in self.quantiles:
            est = self._make_estimator(q)
            est.fit(prepared, y_arr)
            self.models_[q] = est
        self.metadata = ModelMetadata(
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sklearn_version=sklearn.__version__,
            n_samples=int(len(X)),
            n_skus=int(X[self.sku_col].nunique()),
            n_features=int(X.shape[1]),
            config_hash=_config_hash({
                "model": "hgb_quantile",
                "quantiles": list(self.quantiles),
                "max_iter": self.max_iter,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_samples_leaf": self.min_samples_leaf,
            }),
        )

        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retourne un DataFrame avec colonnes ``q_low``, ``q_med``, ``q_high``.

        Les noms sont neutres (low/med/high) plutôt que basés sur la valeur
        numérique du quantile, ce qui permet de changer la config (ex. 5/95
        au lieu de 10/90) sans casser le code en aval.
        """

        if not self.models_:
            raise RuntimeError("Le modèle n'a pas encore été entraîné. Appelez `.fit` d'abord.")
        prepared = self._prepare_features(X)
        sorted_q = sorted(self.models_.keys())

        if len(sorted_q) != 3:
            raise RuntimeError(
                "predict_quantiles() suppose exactement 3 quantiles (low, med, high). "
                f"Configuration actuelle : {sorted_q}"
            )
        out = pd.DataFrame(index=X.index)

        for label, q in zip(QUANTILE_LABELS, sorted_q):
            preds = np.clip(self.models_[q].predict(prepared), a_min=0.0, a_max=None)
            out[label] = preds
        # Garantit la monotonie q_low <= q_med <= q_high (peut etre violee par 3 modeles independants)
        out[list(QUANTILE_LABELS)] = np.sort(out[list(QUANTILE_LABELS)].to_numpy(), axis=1)

        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Raccourci : retourne la prédiction médiane (q_med) sous forme de tableau."""

        return self.predict_quantiles(X)["q_med"].to_numpy()

    def save(self, path: Path | str) -> Path:
        """Sérialise les 3 modèles + un fichier `<path>.meta.json` à côté."""

        if not self.models_:
            raise RuntimeError("Rien à sauvegarder : modèle non entraîné.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"models": self.models_, "feature_cols": self.feature_cols_, "quantiles": self.quantiles},
            target,
        )
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False)

        return target

    @classmethod
    def load(cls, path: Path | str) -> "QuantileGradientBoostingForecaster":
        """Charge un modèle entraîné précédemment via `.save`."""
        target = Path(path)
        bundle = joblib.load(target)
        instance = cls(quantiles=tuple(bundle["quantiles"]))
        instance.models_ = bundle["models"]
        instance.feature_cols_ = bundle["feature_cols"]
        meta_path = target.with_suffix(target.suffix + ".meta.json")

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                instance.metadata = ModelMetadata(**json.load(f))

        return instance


__all__ = [
    "DEFAULT_QUANTILES",
    "ModelMetadata",
    "QUANTILE_LABELS",
    "QuantileGradientBoostingForecaster",
    "RidgeForecaster",
]
