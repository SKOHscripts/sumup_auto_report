"""Tests des modèles quantile et de la simulation Monte-Carlo de rupture."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import features as ft
from stocks.ml.model import QuantileGradientBoostingForecaster
from stocks.ml.projection import (
    _sample_from_quantiles,
    forecast_horizon,
    simulate_rupture,
)


@pytest.fixture
def history_df():
    """Historique long pour permettre l'entraînement HGB (besoin de >= ~30 lignes par SKU)."""
    rng = np.random.default_rng(123)
    rows = []
    base = pd.date_range("2024-06-03", periods=80, freq="W-MON")
    for sku, level in [("chips", 15.0), ("coca", 8.0), ("eau", 25.0)]:
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
    return ds._coerce_dtypes(pd.DataFrame(rows))


# ─── QuantileGradientBoostingForecaster ──────────────────────────────────────


def test_qgbf_predict_quantiles_shape(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=50).fit(X, y)
    out = model.predict_quantiles(X)
    assert set(out.columns) == {"q_low", "q_med", "q_high"}
    assert (out["q_low"] <= out["q_med"]).all()
    assert (out["q_med"] <= out["q_high"]).all()
    assert (out["q_low"] >= 0).all()


def test_qgbf_save_load_roundtrip(tmp_path, history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=30).fit(X, y)
    p = tmp_path / "qgbf.joblib"
    model.save(p)
    assert p.exists()

    loaded = QuantileGradientBoostingForecaster.load(p)
    a = model.predict_quantiles(X).to_numpy()
    b = loaded.predict_quantiles(X).to_numpy()
    np.testing.assert_allclose(a, b)


def test_qgbf_predict_returns_q50_array(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=30).fit(X, y)
    medians = model.predict(X)
    assert medians.shape == (len(X),)


# ─── _sample_from_quantiles ──────────────────────────────────────────────────


def test_sample_quantiles_distribution_is_centered():
    """Avec les fractions par défaut (0.05, 0.95) et q_low=5/q_med=10/q_high=15,
    le 5e percentile des échantillons doit ≈ q_low et le 95e ≈ q_high."""
    rng = np.random.default_rng(0)
    samples = [_sample_from_quantiles(5.0, 10.0, 15.0, rng) for _ in range(5000)]
    arr = np.asarray(samples)
    assert 9.0 < np.median(arr) < 11.0
    assert np.percentile(arr, 5) == pytest.approx(5.0, abs=2.0)
    assert np.percentile(arr, 95) == pytest.approx(15.0, abs=2.0)


def test_sample_quantiles_non_negative():
    rng = np.random.default_rng(0)
    samples = [_sample_from_quantiles(0.0, 0.5, 1.0, rng) for _ in range(1000)]
    assert all(s >= 0.0 for s in samples)


# ─── forecast_horizon ────────────────────────────────────────────────────────


def test_forecast_horizon_shape(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=30).fit(X, y)
    chips_hist = history_df[history_df["stock_sku"] == "chips"]
    fc = forecast_horizon(model, chips_hist, horizon_weeks=8)
    assert len(fc) == 8
    assert set(fc.columns) >= {"q_low", "q_med", "q_high", "week_start", "stock_sku"}
    assert (fc["q_low"] <= fc["q_med"]).all()
    assert (fc["q_med"] <= fc["q_high"]).all()


def test_forecast_horizon_rejects_multi_sku(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=10).fit(X, y)
    with pytest.raises(ValueError, match="mono-SKU"):
        forecast_horizon(model, history_df, horizon_weeks=4)


def test_forecast_horizon_dates_are_consecutive(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=10).fit(X, y)
    chips_hist = history_df[history_df["stock_sku"] == "chips"]
    fc = forecast_horizon(model, chips_hist, horizon_weeks=6)
    diffs = fc["week_start"].apply(lambda d: d.toordinal()).diff().dropna()
    assert (diffs == 7).all()


# ─── simulate_rupture ────────────────────────────────────────────────────────


@pytest.fixture
def quantiles_df():
    """8 semaines de prévisions avec consommation ~10/semaine."""
    base = date(2026, 5, 4)
    return pd.DataFrame([
        {"week_start": base + timedelta(weeks=i), "q_low": 8.0, "q_med": 10.0, "q_high": 12.0}
        for i in range(8)
    ])


def test_simulate_rupture_simple_case(quantiles_df):
    """Stock 50, ~10/semaine → rupture entre semaine 3 et 5."""
    res = simulate_rupture(stock_initial=50.0, weekly_quantiles=quantiles_df, n_simulations=500)
    assert res["prob_rupture"] > 0.99
    assert res["rupture_date_p50"] is not None
    p50 = res["rupture_date_p50"]
    assert date(2026, 5, 18) <= p50 <= date(2026, 6, 8)
    assert res["rupture_date_p10"] <= res["rupture_date_p50"] <= res["rupture_date_p90"]


def test_simulate_rupture_no_rupture_in_horizon(quantiles_df):
    """Stock 1000 sur 8 semaines × ~10 → jamais de rupture."""
    res = simulate_rupture(stock_initial=1000.0, weekly_quantiles=quantiles_df, n_simulations=200)
    assert res["prob_rupture"] == 0.0
    assert res["rupture_date_p50"] is None


def test_simulate_rupture_with_incoming(quantiles_df):
    """Stock initial 30, +50 à la semaine 3 → rupture repoussée."""
    res_no_incoming = simulate_rupture(
        stock_initial=30.0, weekly_quantiles=quantiles_df, n_simulations=300,
    )
    res_with_incoming = simulate_rupture(
        stock_initial=30.0,
        weekly_quantiles=quantiles_df,
        incoming_qty=50.0,
        incoming_eta=date(2026, 5, 25),
        n_simulations=300,
    )
    if res_no_incoming["rupture_date_p50"] and res_with_incoming["rupture_date_p50"]:
        assert res_with_incoming["rupture_date_p50"] >= res_no_incoming["rupture_date_p50"]


def test_simulate_rupture_reproducible(quantiles_df):
    a = simulate_rupture(stock_initial=50.0, weekly_quantiles=quantiles_df, n_simulations=200, seed=42)
    b = simulate_rupture(stock_initial=50.0, weekly_quantiles=quantiles_df, n_simulations=200, seed=42)
    assert a["rupture_date_p50"] == b["rupture_date_p50"]
    np.testing.assert_array_equal(a["trajectories"], b["trajectories"])


def test_simulate_rupture_empty_quantiles_raises():
    with pytest.raises(ValueError, match="vide"):
        simulate_rupture(stock_initial=10.0, weekly_quantiles=pd.DataFrame())


# ─── Integration : forecast + simulate ───────────────────────────────────────


def test_forecast_then_simulate_end_to_end(history_df):
    X, y, _ = ft.prepare_training_table(history_df)
    model = QuantileGradientBoostingForecaster(max_iter=50).fit(X, y)
    chips_hist = history_df[history_df["stock_sku"] == "chips"]
    fc = forecast_horizon(model, chips_hist, horizon_weeks=12)
    res = simulate_rupture(stock_initial=80.0, weekly_quantiles=fc, n_simulations=300)
    # Avec ~15/semaine de chips et 80 unités, rupture attendue dans l'horizon
    assert res["rupture_date_p50"] is not None
    assert res["prob_rupture"] > 0.5
