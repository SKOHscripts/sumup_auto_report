"""Tests du modèle baseline Ridge."""
import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import features as ft
from stocks.ml.model import RidgeForecaster


@pytest.fixture
def history_df():
    """50 semaines pour 3 SKU avec un signal exploitable (lag-1 fortement prédictif)."""
    rng = np.random.default_rng(42)
    rows = []
    base = pd.date_range("2025-01-06", periods=50, freq="W-MON")
    for sku, base_level in [("chips", 12.0), ("coca", 6.0), ("eau", 20.0)]:
        prev = base_level
        for ts in base:
            iso = ts.isocalendar()
            usage = max(0.0, 0.7 * prev + 0.3 * base_level + rng.normal(0, 0.5))
            rows.append({
                "stock_sku": sku,
                "week_label": f"{iso.year}-W{iso.week:02d}",
                "year": iso.year,
                "week": iso.week,
                "week_start": ts.date(),
                "usage": usage,
                "sales_count": int(usage),
            })
            prev = usage
    return ds._coerce_dtypes(pd.DataFrame(rows))


def test_fit_predict_pipeline(history_df):
    X, y, _meta = ft.prepare_training_table(history_df)
    model = RidgeForecaster(alpha=1.0).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert (preds >= 0).all(), "Les prédictions ne doivent pas être négatives (clip)"


def test_fit_then_predict_unseen_split(history_df):
    """Train sur 80% temporel, prédit sur 20%, vérifie une amélioration vs la moyenne globale."""
    X, y, meta = ft.prepare_training_table(history_df)
    sorted_idx = meta.sort_values(["year", "week"]).index
    cut = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:cut]
    test_idx = sorted_idx[cut:]

    model = RidgeForecaster(alpha=1.0).fit(X.loc[train_idx], y.loc[train_idx])
    preds = model.predict(X.loc[test_idx])

    mae_model = float(np.mean(np.abs(preds - y.loc[test_idx].values)))
    mae_global = float(np.mean(np.abs(y.loc[train_idx].mean() - y.loc[test_idx].values)))
    assert mae_model < mae_global, f"Ridge ({mae_model:.2f}) devrait battre la moyenne globale ({mae_global:.2f})"


def test_predict_before_fit_raises():
    model = RidgeForecaster()
    with pytest.raises(RuntimeError, match="entraîné"):
        model.predict(pd.DataFrame({"stock_sku": ["chips"]}))


def test_save_before_fit_raises(tmp_path):
    model = RidgeForecaster()
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "model.joblib")


def test_save_then_load_roundtrip(tmp_path, history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = RidgeForecaster(alpha=0.5).fit(X, y)
    model_path = tmp_path / "ridge.joblib"
    model.save(model_path)

    assert model_path.exists()
    assert model_path.with_suffix(model_path.suffix + ".meta.json").exists()

    loaded = RidgeForecaster.load(model_path)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))
    assert loaded.metadata.n_samples == model.metadata.n_samples
    assert loaded.metadata.n_skus == 3


def test_metadata_populated_after_fit(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = RidgeForecaster(alpha=1.0).fit(X, y)
    assert model.metadata.trained_at != ""
    assert model.metadata.n_samples == len(X)
    assert model.metadata.n_skus == 3
    assert model.metadata.sklearn_version != ""
    assert len(model.metadata.config_hash) == 12


def test_handles_unknown_sku_in_predict(history_df):
    """L'OneHotEncoder doit ignorer les SKU inconnus à la prédiction (handle_unknown=ignore)."""
    X, y, _ = ft.prepare_training_table(history_df)
    model = RidgeForecaster(alpha=1.0).fit(X, y)
    X_new = X.iloc[[0]].copy()
    X_new["stock_sku"] = "sku_inconnu"
    preds = model.predict(X_new)
    assert len(preds) == 1
    assert preds[0] >= 0
