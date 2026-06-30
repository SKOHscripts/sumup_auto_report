"""Tests du module d'évaluation walk-forward."""
import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import evaluation as ev


@pytest.fixture
def long_history():
    """100 semaines pour 2 SKU avec un signal autoregressif modéré."""
    rng = np.random.default_rng(11)
    rows = []
    base = pd.date_range("2024-01-01", periods=100, freq="W-MON")
    for sku, level in [("chips", 12.0), ("coca", 6.0)]:
        prev = level
        for ts in base:
            iso = ts.isocalendar()
            usage = max(0.0, 0.5 * prev + 0.5 * level + rng.normal(0, 0.8))
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


def test_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.5, 3.5])
    assert ev.mae(y, p) == pytest.approx(0.5)


def test_mape_handles_zero():
    y = np.array([0.0, 0.0, 1.0])
    p = np.array([0.0, 1.0, 1.0])
    val = ev.mape(y, p, eps=1.0)
    assert val == pytest.approx((0 + 1.0 + 0) / 3)


def test_pinball_loss_q50_equals_mae_half():
    y = np.array([10.0, 12.0, 8.0])
    p = np.array([11.0, 11.0, 9.0])
    expected = float(np.mean(np.abs(y - p))) / 2
    assert ev.pinball_loss(y, p, 0.5) == pytest.approx(expected)


def test_coverage_full():
    y = np.array([5.0, 6.0, 7.0])
    lo = np.array([4.0, 5.0, 6.0])
    hi = np.array([6.0, 7.0, 8.0])
    assert ev.coverage(y, lo, hi) == 1.0


def test_coverage_partial():
    y = np.array([5.0, 10.0, 7.0])
    lo = np.array([4.0, 5.0, 6.0])
    hi = np.array([6.0, 7.0, 8.0])
    assert ev.coverage(y, lo, hi) == pytest.approx(2 / 3)


def test_walk_forward_backtest_runs(long_history):
    metrics = ev.walk_forward_backtest(long_history, n_folds=3, max_iter=30, min_train_size=80)
    assert metrics.n_folds > 0
    assert 0.0 <= metrics.mape <= 5.0
    assert 0.0 <= metrics.coverage_band <= 1.0
    assert len(metrics.fold_metrics) == metrics.n_folds


def test_walk_forward_returns_empty_when_too_short():
    tiny = pd.DataFrame({
        "stock_sku": ["chips"] * 10,
        "week_label": [f"2026-W{i:02d}" for i in range(1, 11)],
        "year": [2026] * 10,
        "week": list(range(1, 11)),
        "week_start": pd.date_range("2026-01-05", periods=10, freq="W-MON").date,
        "usage": [10.0] * 10,
        "sales_count": [10] * 10,
    })
    tiny = ds._coerce_dtypes(tiny)
    metrics = ev.walk_forward_backtest(tiny, min_train_size=200)
    assert metrics.n_folds == 0


def test_baseline_avg_rolling4(long_history):
    val = ev.baseline_avg_rolling4(long_history)
    assert 0.0 < val < 5.0


def test_is_model_promotable_passes():
    metrics = ev.EvaluationMetrics(
        mae=1.0, mape=0.20, pinball_low=0.5, pinball_med=0.5, pinball_high=0.5,
        coverage_band=0.78, n_samples=100, n_folds=5,
    )
    promotable, reasons = ev.is_model_promotable(metrics, baseline_mape=0.30)
    assert promotable is True
    assert reasons == []


def test_is_model_promotable_rejects_high_mape():
    metrics = ev.EvaluationMetrics(
        mae=10.0, mape=0.60, pinball_low=2.0, pinball_med=3.0, pinball_high=2.0,
        coverage_band=0.80, n_samples=100, n_folds=5,
    )
    promotable, reasons = ev.is_model_promotable(metrics, baseline_mape=0.50)
    assert promotable is False
    assert any("MAPE" in r for r in reasons)


def test_is_model_promotable_rejects_bad_coverage():
    metrics = ev.EvaluationMetrics(
        mae=1.0, mape=0.20, pinball_low=0.5, pinball_med=0.5, pinball_high=0.5,
        coverage_band=0.50, n_samples=100, n_folds=5,
    )
    promotable, reasons = ev.is_model_promotable(metrics, baseline_mape=0.30)
    assert promotable is False
    assert any("Coverage" in r for r in reasons)


def test_is_model_promotable_rejects_when_worse_than_baseline():
    metrics = ev.EvaluationMetrics(
        mae=1.0, mape=0.30, pinball_low=0.5, pinball_med=0.5, pinball_high=0.5,
        coverage_band=0.80, n_samples=100, n_folds=5,
    )
    promotable, reasons = ev.is_model_promotable(metrics, baseline_mape=0.20)
    assert promotable is False
    assert any("baseline" in r.lower() for r in reasons)


def test_is_model_promotable_within_relative_margin():
    """ML légèrement moins bon que la baseline mais dans la marge → promu.

    Reflète la politique « demande intermittente » : MAPE absolue inatteignable,
    on accepte un modèle ~= baseline (à la marge près) pour ses intervalles.
    """
    metrics = ev.EvaluationMetrics(
        mae=9.0, mape=0.73, pinball_low=2.0, pinball_med=3.0, pinball_high=2.0,
        coverage_band=0.69, n_samples=200, n_folds=5,
    )
    promotable, reasons = ev.is_model_promotable(
        metrics, baseline_mape=0.70, relative_mape_margin=0.10,
    )
    assert promotable is True  # 0.73 <= 0.70 * 1.10 = 0.77 et coverage dans 80±15
    assert reasons == []


def test_is_model_promotable_uses_absolute_threshold_without_baseline():
    """Sans baseline, on retombe sur le seuil MAPE absolu."""
    metrics = ev.EvaluationMetrics(
        mae=9.0, mape=0.73, pinball_low=2.0, pinball_med=3.0, pinball_high=2.0,
        coverage_band=0.80, n_samples=200, n_folds=5,
    )
    promotable, reasons = ev.is_model_promotable(metrics, baseline_mape=None)
    assert promotable is False
    assert any("MAPE" in r for r in reasons)


def test_is_model_promotable_no_folds():
    metrics = ev.EvaluationMetrics(n_folds=0)
    promotable, reasons = ev.is_model_promotable(metrics)
    assert promotable is False
    assert reasons == ["aucun fold valide"]
