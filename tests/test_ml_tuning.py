"""Tests du tuning d'hyperparamètres."""
import numpy as np
import pandas as pd
import pytest

from stocks.ml import config as cfg_mod
from stocks.ml import dataset as ds
from stocks.ml import tuning


@pytest.fixture
def long_history():
    """Historique long pour permettre TimeSeriesSplit avec 4 plis."""
    rng = np.random.default_rng(42)
    rows = []
    base = pd.date_range("2024-01-01", periods=80, freq="W-MON")
    for sku, level in [("chips", 10.0), ("coca", 5.0), ("eau", 20.0)]:
        prev = level
        for ts in base:
            iso = ts.isocalendar()
            usage = max(0.0, 0.6 * prev + 0.4 * level + rng.normal(0, 0.5))
            rows.append({"stock_sku": sku, "week_label": f"{iso.year}-W{iso.week:02d}",
                         "year": iso.year, "week": iso.week, "week_start": ts.date(),
                         "usage": usage, "sales_count": int(usage)})
            prev = usage
    return ds._coerce_dtypes(pd.DataFrame(rows))


def test_tune_hyperparameters_returns_best_params(long_history):
    """Vérifie qu'un tuning court trouve un dict de paramètres + un score."""
    grid = {
        "max_iter": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.05],
        "min_samples_leaf": [5, 10],
    }
    best, score = tuning.tune_hyperparameters(long_history, n_iter=4, n_splits=3, grid=grid)
    assert "max_iter" in best
    assert "max_depth" in best
    assert "learning_rate" in best
    assert "min_samples_leaf" in best
    assert score >= 0  # pinball loss positive


def test_tune_and_save_persists_config_when_better(tmp_path, monkeypatch, long_history):
    """tune_and_save écrit la nouvelle config si elle améliore le backtest."""
    target = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", target)

    grid = {
        "max_iter": [50],
        "max_depth": [3],
        "learning_rate": [0.05],
        "min_samples_leaf": [5],
    }
    monkeypatch.setattr(tuning, "PARAM_GRID", grid)
    # Candidate (max_iter=50) meilleur que la config courante (défaut 200).
    monkeypatch.setattr(
        tuning, "_backtest_mape",
        lambda hist, cfg, params: 0.4 if params.get("max_iter") == 50 else 0.9,
    )
    cfg = tuning.tune_and_save(long_history, n_iter_coarse=2, n_iter_fine=2)

    assert target.exists()
    assert cfg.tuned_at is not None
    assert cfg.tuned_params["max_iter"] == 50

    reloaded = cfg_mod.load_config(target)
    assert reloaded.tuned_params["max_iter"] == 50


def test_tune_and_save_keeps_config_when_not_better(tmp_path, monkeypatch, long_history):
    """Garde-fou : si le tuning ne bat pas la config actuelle, on ne change rien."""
    target = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", target)

    grid = {"max_iter": [50], "max_depth": [3], "learning_rate": [0.05], "min_samples_leaf": [5]}
    monkeypatch.setattr(tuning, "PARAM_GRID", grid)
    # Candidate (max_iter=50) PIRE que la config courante (défaut 200).
    monkeypatch.setattr(
        tuning, "_backtest_mape",
        lambda hist, cfg, params: 0.9 if params.get("max_iter") == 50 else 0.4,
    )
    cfg = tuning.tune_and_save(long_history, n_iter_coarse=2, n_iter_fine=2)

    # Config inchangée (params par défaut conservés) et fichier non réécrit.
    assert cfg.tuned_params["max_iter"] == cfg_mod.DEFAULT_HGB_PARAMS["max_iter"]
    assert not target.exists()


def test_pinball_score_is_negative_for_minimization():
    """Le scorer doit etre negatif (sklearn maximise)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "stock_sku": ["x"] * 5})
    X["stock_sku"] = X["stock_sku"].astype("category")
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    est = HistGradientBoostingRegressor(loss="quantile", quantile=0.5,
                                        categorical_features=["stock_sku"], max_iter=10)
    est.fit(X, y)
    scorer = tuning._pinball_score(0.5)
    score = scorer(est, X, y)
    assert score <= 0  # pinball loss negatif sous convention sklearn
