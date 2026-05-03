"""Tests du diagnostic par SKU."""
import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import diagnose as dg


@pytest.fixture
def mixed_history():
    """Mix de SKU : un régulier, un sporadique, un volatil."""
    rows = []
    base = pd.date_range("2025-12-01", periods=20, freq="W-MON")
    rng = np.random.default_rng(0)

    for i, ts in enumerate(base):
        iso = ts.isocalendar()
        rows.append({"stock_sku": "regulier", "week_label": f"{iso.year}-W{iso.week:02d}",
                     "year": iso.year, "week": iso.week, "week_start": ts.date(),
                     "usage": 10.0 + rng.normal(0, 0.5), "sales_count": 10})
        rows.append({"stock_sku": "sporadique", "week_label": f"{iso.year}-W{iso.week:02d}",
                     "year": iso.year, "week": iso.week, "week_start": ts.date(),
                     "usage": 5.0 if i % 4 == 0 else 0.0, "sales_count": 5 if i % 4 == 0 else 0})
        rows.append({"stock_sku": "volatil", "week_label": f"{iso.year}-W{iso.week:02d}",
                     "year": iso.year, "week": iso.week, "week_start": ts.date(),
                     "usage": float(max(0, rng.normal(8, 6))), "sales_count": 8})
    return ds._coerce_dtypes(pd.DataFrame(rows))


def test_diagnose_returns_one_row_per_sku(mixed_history):
    df = dg.diagnose(mixed_history)
    assert len(df) == 3
    assert set(df["stock_sku"]) == {"regulier", "sporadique", "volatil"}


def test_diagnose_fields_present(mixed_history):
    df = dg.diagnose(mixed_history)
    expected = {
        "stock_sku", "n_weeks", "n_zeros", "pct_zeros", "mean_usage",
        "std_usage", "cv", "last_4w_mean", "mape_naive", "mape_avg4",
    }
    assert set(df.columns) == expected


def test_diagnose_counts_zeros(mixed_history):
    df = dg.diagnose(mixed_history)
    sporadique = df[df["stock_sku"] == "sporadique"].iloc[0]
    assert sporadique["n_zeros"] == 15  # 20 semaines, 5 non-nulles (i=0,4,8,12,16)
    assert sporadique["pct_zeros"] == pytest.approx(0.75)


def test_diagnose_volatility(mixed_history):
    df = dg.diagnose(mixed_history)
    regulier = df[df["stock_sku"] == "regulier"].iloc[0]
    volatil = df[df["stock_sku"] == "volatil"].iloc[0]
    # Le SKU volatil doit avoir un CV plus elevé que le régulier
    assert volatil["cv"] > regulier["cv"]


def test_diagnose_sorted_by_mape_desc(mixed_history):
    df = dg.diagnose(mixed_history)
    # mape_avg4 doit etre decroissante (NaN en dernier)
    valid = df.dropna(subset=["mape_avg4"])
    assert (valid["mape_avg4"].diff().dropna() <= 0).all()


def test_diagnose_empty_history():
    empty = ds._coerce_dtypes(pd.DataFrame(columns=ds.SCHEMA_COLUMNS))
    df = dg.diagnose(empty)
    assert len(df) == 0


def test_format_table_runs(mixed_history):
    df = dg.diagnose(mixed_history)
    text = dg.format_table(df)
    assert "regulier" in text
    assert "sporadique" in text
    assert "volatil" in text


def test_format_table_top_n(mixed_history):
    df = dg.diagnose(mixed_history)
    text = dg.format_table(df, top_n=1)
    lines = text.split("\n")
    assert len(lines) == 3  # header + separator + 1 ligne


def test_format_table_handles_empty():
    empty = pd.DataFrame()
    assert "(aucun" in dg.format_table(empty)


def test_save_csv_writes_file(tmp_path, mixed_history):
    df = dg.diagnose(mixed_history)
    target = tmp_path / "diag.csv"
    dg.save_csv(df, target)
    assert target.exists()
    reloaded = pd.read_csv(target)
    assert len(reloaded) == 3
