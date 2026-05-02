"""Test d'intégration : la persistance ML doit être appelée par run_stock_report
sans bloquer le pipeline en cas de souci, et alimenter le parquet."""
from unittest.mock import patch

import pandas as pd

import stocks.sumup_stocks as sumup_stocks


def test_persist_weekly_history_writes_parquet(tmp_path, monkeypatch):
    target = tmp_path / "weekly_usage.parquet"
    monkeypatch.setattr("stocks.ml.dataset.DATASET_PATH", target)

    weekly_usage = {"chips": {"2026-W18": 10.0}, "coca": {"2026-W18": 4.0}}
    weekly_sales_count = {"chips": {"2026-W18": 10}, "coca": {"2026-W18": 4}}

    with patch("stocks.ml.dataset.update_weekly_usage") as mock_update:
        mock_update.return_value = pd.DataFrame(
            {"stock_sku": ["chips", "coca"], "week_label": ["2026-W18", "2026-W18"]}
        )
        sumup_stocks._persist_weekly_history(weekly_usage, weekly_sales_count)
        mock_update.assert_called_once_with(weekly_usage, weekly_sales_count)


def test_persist_weekly_history_handles_failure(caplog):
    """Si update_weekly_usage lève, on log un warning mais on ne crashe pas."""
    weekly_usage = {"chips": {"2026-W18": 10.0}}
    weekly_sales_count = {"chips": {"2026-W18": 10}}

    with patch("stocks.ml.dataset.update_weekly_usage", side_effect=RuntimeError("disk full")):
        with caplog.at_level("WARNING"):
            sumup_stocks._persist_weekly_history(weekly_usage, weekly_sales_count)
    assert any("non bloquant" in rec.message for rec in caplog.records)


def test_persist_weekly_history_handles_missing_dependency(caplog):
    """Si pandas/pyarrow ne sont pas disponibles, on doit logger info et sortir."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "stocks.ml.dataset":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with caplog.at_level("INFO"):
            sumup_stocks._persist_weekly_history({}, {})
    assert any("Persistance ML ignoree" in rec.message for rec in caplog.records)


def test_persist_end_to_end_via_dataset(tmp_path, monkeypatch):
    """Bout en bout sans mock : l'appel doit produire un parquet lisible."""
    target = tmp_path / "weekly_usage.parquet"
    monkeypatch.setattr("stocks.ml.dataset.DATASET_PATH", target)

    weekly_usage = {"chips": {"2026-W17": 5.0, "2026-W18": 7.0}}
    weekly_sales_count = {"chips": {"2026-W17": 5, "2026-W18": 7}}

    sumup_stocks._persist_weekly_history(weekly_usage, weekly_sales_count)

    assert target.exists()
    df = pd.read_parquet(target)
    assert len(df) == 2
    assert set(df["week_label"]) == {"2026-W17", "2026-W18"}
