"""Tests du registre des modèles ML."""
import csv

import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import features as ft
from stocks.ml import registry as reg
from stocks.ml.evaluation import EvaluationMetrics
from stocks.ml.model import QuantileGradientBoostingForecaster


@pytest.fixture
def trained_model():
    rng = np.random.default_rng(0)
    rows = []
    base = pd.date_range("2025-01-06", periods=60, freq="W-MON")
    for sku in ("chips", "coca"):
        for ts in base:
            iso = ts.isocalendar()
            rows.append({
                "stock_sku": sku,
                "week_label": f"{iso.year}-W{iso.week:02d}",
                "year": iso.year,
                "week": iso.week,
                "week_start": ts.date(),
                "usage": float(max(0.0, rng.normal(10, 2))),
                "sales_count": 10,
            })
    history = ds._coerce_dtypes(pd.DataFrame(rows))
    X, y, _ = ft.prepare_training_table(history)
    return QuantileGradientBoostingForecaster(max_iter=20).fit(X, y)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(reg, "ARCHIVE_DIR", tmp_path / "models" / "archive")
    monkeypatch.setattr(reg, "HISTORY_CSV", tmp_path / "models" / "history.csv")
    monkeypatch.setattr(reg, "CURRENT_MODEL", tmp_path / "models" / "current.joblib")
    yield


def test_archive_creates_files(trained_model):
    p = reg.archive_model(trained_model, week_label="2026-W18")
    assert p.exists()
    assert p.parent.name == "2026_W18"
    meta = p.with_suffix(p.suffix + ".meta.json")
    assert meta.exists()


def test_set_current_points_to_archive(trained_model):
    p = reg.archive_model(trained_model, week_label="2026-W18")
    reg.set_current(p)
    assert reg.CURRENT_MODEL.exists()
    loaded = reg.load_current()
    assert loaded is not None
    assert loaded.metadata.config_hash == trained_model.metadata.config_hash


def test_load_current_returns_none_when_absent():
    assert reg.load_current() is None


def test_promote_if_better_records_history(trained_model):
    metrics = EvaluationMetrics(
        mae=1.0, mape=0.25, pinball_low=0.5, pinball_med=0.5, pinball_high=0.5,
        coverage_band=0.80, n_samples=100, n_folds=5,
    )
    promoted, archive_path = reg.promote_if_better(
        model=trained_model,
        metrics=metrics,
        promotable=True,
        reasons=[],
        baseline_mape=0.30,
        week_label="2026-W18",
    )
    assert promoted is True
    assert archive_path.exists()
    assert reg.CURRENT_MODEL.exists()
    assert reg.HISTORY_CSV.exists()
    rows = reg.recent_history()
    assert len(rows) == 1
    assert rows[0]["promoted"] == "1"
    assert rows[0]["week_label"] == "2026-W18"


def test_promote_if_better_archives_even_when_not_promotable(trained_model):
    metrics = EvaluationMetrics(
        mae=10.0, mape=0.60, pinball_low=2.0, pinball_med=3.0, pinball_high=2.0,
        coverage_band=0.50, n_samples=100, n_folds=5,
    )
    promoted, archive_path = reg.promote_if_better(
        model=trained_model,
        metrics=metrics,
        promotable=False,
        reasons=["MAPE trop eleve", "Coverage hors cible"],
        week_label="2026-W19",
    )
    assert promoted is False
    assert archive_path.exists()
    # current.joblib n'existe PAS car non promu
    assert not reg.CURRENT_MODEL.exists()
    rows = reg.recent_history()
    assert rows[0]["promoted"] == "0"
    assert "MAPE" in rows[0]["reasons"]


def test_history_csv_has_correct_header(trained_model):
    metrics = EvaluationMetrics(n_folds=5, mape=0.2, coverage_band=0.8)
    reg.promote_if_better(trained_model, metrics, True, [], 0.3, "2026-W18")
    with open(reg.HISTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == reg.HISTORY_HEADER


def test_recent_history_empty_when_no_journal():
    assert reg.recent_history() == []


def test_detect_drift_no_history():
    drifted, _msg = reg.detect_drift(n=3)
    assert drifted is False


def test_detect_drift_triggers_after_3_bad_weeks(trained_model):
    bad = EvaluationMetrics(n_folds=5, mape=0.6, coverage_band=0.5)
    for week in ("2026-W16", "2026-W17", "2026-W18"):
        reg.promote_if_better(trained_model, bad, False, ["MAPE trop eleve"], None, week)
    drifted, msg = reg.detect_drift(n=3, mape_threshold=0.45)
    assert drifted is True
    assert "MAPE" in msg


def test_detect_drift_silent_when_only_2_bad(trained_model):
    bad = EvaluationMetrics(n_folds=5, mape=0.6, coverage_band=0.5)
    good = EvaluationMetrics(n_folds=5, mape=0.2, coverage_band=0.8)
    reg.promote_if_better(trained_model, bad, False, [], None, "2026-W16")
    reg.promote_if_better(trained_model, good, True, [], None, "2026-W17")
    reg.promote_if_better(trained_model, bad, False, [], None, "2026-W18")
    drifted, _msg = reg.detect_drift(n=3, mape_threshold=0.45)
    assert drifted is False
