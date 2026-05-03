"""Tests de l'orchestrateur ML (stocks/ml/inference.py) et du wiring CLI."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import inference as inf


@pytest.fixture
def long_history(tmp_path):
    """Historique long persistant en parquet pour les tests d'inférence."""
    rng = np.random.default_rng(7)
    rows = []
    base = pd.date_range("2024-09-02", periods=70, freq="W-MON")
    for sku, level in [("chips", 12.0), ("coca", 6.0)]:
        prev = level
        for ts in base:
            iso = ts.isocalendar()
            usage = max(0.0, 0.6 * prev + 0.4 * level + rng.normal(0, 1.0))
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
    df = ds._coerce_dtypes(pd.DataFrame(rows))
    parquet = tmp_path / "weekly_usage.parquet"
    ds.save_weekly_usage(df, parquet)
    return parquet, df


def test_train_global_model_short_history_returns_none():
    short = pd.DataFrame({
        "stock_sku": ["x"], "week_label": ["2026-W01"], "year": [2026], "week": [1],
        "week_start": [date(2026, 1, 5)], "usage": [1.0], "sales_count": [1],
    })
    short = ds._coerce_dtypes(short)
    assert inf.train_global_model(short) is None


def test_train_global_model_succeeds(long_history):
    _, df = long_history
    model = inf.train_global_model(df, max_iter=30)
    assert model is not None
    assert model.metadata.n_skus == 2


def test_project_for_sku_returns_dict(long_history):
    _, df = long_history
    model = inf.train_global_model(df, max_iter=30)
    proj = inf.project_for_sku(model, df, sku="chips", stock_initial=100.0, horizon_weeks=8, n_simulations=200)
    assert proj is not None
    assert "rupture_date_med" in proj
    assert "weekly_forecast" in proj
    assert len(proj["weekly_forecast"]) == 8
    assert "model_version" in proj


def test_project_for_sku_short_history_returns_none(long_history):
    _, df = long_history
    model = inf.train_global_model(df, max_iter=30)
    # SKU avec quelques semaines seulement
    short = df[df["stock_sku"] == "chips"].head(5)
    proj = inf.project_for_sku(model, short, sku="chips", stock_initial=100.0)
    assert proj is None


def test_attach_ml_projections_enriches_kpis(long_history):
    parquet, _ = long_history
    kpis = [
        {"stock_sku": "chips", "available_stock": 80.0, "incoming_qty": 0.0},
        {"stock_sku": "coca", "available_stock": 40.0, "incoming_qty": 0.0},
    ]
    out = inf.attach_ml_projections(kpis, history_path=parquet)
    assert "ml_projection" in out[0]
    assert "ml_projection" in out[1]
    assert out[0]["ml_projection"]["weekly_forecast"]


def test_attach_ml_projections_no_history_returns_unchanged(tmp_path):
    """Pas de parquet -> les KPIs sont retournés inchangés."""
    kpis = [{"stock_sku": "chips", "available_stock": 80.0}]
    out = inf.attach_ml_projections(kpis, history_path=tmp_path / "absent.parquet")
    assert "ml_projection" not in out[0]


def test_attach_ml_projections_unknown_sku_skipped(long_history):
    parquet, _ = long_history
    kpis = [{"stock_sku": "sku_inexistant", "available_stock": 10.0}]
    out = inf.attach_ml_projections(kpis, history_path=parquet)
    # Le SKU n'a pas d'historique -> pas de projection ML attachée
    assert "ml_projection" not in out[0]


def test_parse_eta_handles_invalid():
    assert inf._parse_eta(None) is None
    assert inf._parse_eta("") is None
    assert inf._parse_eta("not-a-date") is None
    assert inf._parse_eta("2026-06-15") == date(2026, 6, 15)
    assert inf._parse_eta("2026-06-15T10:00:00") == date(2026, 6, 15)
