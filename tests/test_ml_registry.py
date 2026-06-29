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


# ── Migration de schéma history.csv (dérive d'en-tête) ────────────────────────

def _write_history(path, header, rows):
    """Écrit un history.csv brut avec l'en-tête et les lignes fournis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _legacy_row(week, mape):
    # 14 colonnes : pas de rmse ni mean_bias
    return ["2026-05-04T00:00:00+00:00", week, "v1", "0", "8.0", f"{mape:.4f}",
            "1.0", "2.0", "1.5", "0.64", "0.67", "95", "5", "MAPE"]


def _current_row(week, mape):
    # 16 colonnes : rmse et mean_bias insérés
    return ["2026-05-18T00:00:00+00:00", week, "v2", "0", "7.5", "17.6",
            f"{mape:.4f}", "1.2", "2.0", "3.7", "2.7", "0.51", "0.68", "95", "5", "MAPE"]


def test_recent_history_realigns_drifted_header():
    """Avec un en-tête à l'ancien schéma, les lignes 16 colonnes sont relues alignées."""
    _write_history(reg.HISTORY_CSV, reg.LEGACY_HISTORY_HEADER,
                   [_legacy_row("2026-W18", 0.8376), _current_row("2026-W21", 0.7018)])
    rows = reg.recent_history(n=10)
    assert len(rows) == 2
    # La MAPE doit être la vraie valeur, pas le rmse décalé.
    assert float(rows[0]["mape"]) == pytest.approx(0.8376)
    assert float(rows[1]["mape"]) == pytest.approx(0.7018)
    assert rows[0]["rmse"] == ""          # ligne legacy : rmse vide
    assert float(rows[1]["rmse"]) == pytest.approx(17.6)


def test_migrate_history_file_rewrites_header():
    _write_history(reg.HISTORY_CSV, reg.LEGACY_HISTORY_HEADER,
                   [_legacy_row("2026-W18", 0.8376), _current_row("2026-W21", 0.7018)])
    assert reg.migrate_history_file() is True
    with open(reg.HISTORY_CSV, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == reg.HISTORY_HEADER
    # Idempotent : déjà au bon schéma → pas de seconde migration.
    assert reg.migrate_history_file() is False


def test_append_history_autoheals_drifted_header(trained_model):
    _write_history(reg.HISTORY_CSV, reg.LEGACY_HISTORY_HEADER, [_legacy_row("2026-W18", 0.8376)])
    metrics = EvaluationMetrics(n_folds=5, mae=7.0, rmse=20.0, mape=0.55,
                                mean_bias=1.0, coverage_band=0.6, n_samples=100)
    reg.append_history(metrics, promoted=False, version="v3", week_label="2026-W22",
                       baseline_mape=0.70, reasons=["test"])
    with open(reg.HISTORY_CSV, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == reg.HISTORY_HEADER
    rows = reg.recent_history(n=10)
    assert float(rows[-1]["mape"]) == pytest.approx(0.55)
    assert float(rows[0]["mape"]) == pytest.approx(0.8376)
