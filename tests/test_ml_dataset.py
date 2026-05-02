"""Tests de la persistance hebdomadaire (stocks/ml/dataset.py)."""
from datetime import date

import pandas as pd
import pytest

from stocks.ml import dataset as ds


@pytest.fixture
def tmp_parquet(tmp_path):
    return tmp_path / "weekly_usage.parquet"


@pytest.fixture
def sample_weekly_usage():
    return {
        "chips": {"2026-W10": 12.5, "2026-W11": 8.0},
        "coca": {"2026-W10": 4.0, "2026-W11": 6.5, "2026-W12": 3.0},
    }


@pytest.fixture
def sample_sales_count():
    return {
        "chips": {"2026-W10": 12, "2026-W11": 8},
        "coca": {"2026-W10": 4, "2026-W11": 7, "2026-W12": 3},
    }


def test_load_returns_empty_when_file_missing(tmp_parquet):
    df = ds.load_weekly_usage(tmp_parquet)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ds.SCHEMA_COLUMNS


def test_dict_to_dataframe_schema(sample_weekly_usage, sample_sales_count):
    df = ds.weekly_usage_dict_to_dataframe(sample_weekly_usage, sample_sales_count)
    assert list(df.columns) == ds.SCHEMA_COLUMNS
    assert len(df) == 5
    assert set(df["stock_sku"].unique()) == {"chips", "coca"}
    chips_w10 = df[(df["stock_sku"] == "chips") & (df["week_label"] == "2026-W10")].iloc[0]
    assert chips_w10["usage"] == 12.5
    assert chips_w10["sales_count"] == 12
    assert chips_w10["year"] == 2026
    assert chips_w10["week"] == 10
    assert chips_w10["week_start"] == date(2026, 3, 2)


def test_dict_to_dataframe_without_sales_count(sample_weekly_usage):
    df = ds.weekly_usage_dict_to_dataframe(sample_weekly_usage, None)
    assert (df["sales_count"] == 0).all()


def test_save_then_load_roundtrip(tmp_parquet, sample_weekly_usage, sample_sales_count):
    df = ds.weekly_usage_dict_to_dataframe(sample_weekly_usage, sample_sales_count)
    ds.save_weekly_usage(df, tmp_parquet)
    assert tmp_parquet.exists()

    reloaded = ds.load_weekly_usage(tmp_parquet)
    assert len(reloaded) == len(df)
    assert set(reloaded["stock_sku"].unique()) == {"chips", "coca"}
    assert reloaded["usage"].sum() == pytest.approx(df["usage"].sum())


def test_update_creates_file(tmp_parquet, sample_weekly_usage, sample_sales_count):
    merged = ds.update_weekly_usage(sample_weekly_usage, sample_sales_count, tmp_parquet)
    assert tmp_parquet.exists()
    assert len(merged) == 5

    reloaded = ds.load_weekly_usage(tmp_parquet)
    assert len(reloaded) == 5


def test_update_is_idempotent(tmp_parquet, sample_weekly_usage, sample_sales_count):
    ds.update_weekly_usage(sample_weekly_usage, sample_sales_count, tmp_parquet)
    merged = ds.update_weekly_usage(sample_weekly_usage, sample_sales_count, tmp_parquet)
    assert len(merged) == 5  # pas de doublon


def test_update_overwrites_existing_week(tmp_parquet, sample_weekly_usage, sample_sales_count):
    ds.update_weekly_usage(sample_weekly_usage, sample_sales_count, tmp_parquet)

    corrected = {"chips": {"2026-W10": 99.0}}
    corrected_sales = {"chips": {"2026-W10": 99}}
    merged = ds.update_weekly_usage(corrected, corrected_sales, tmp_parquet)

    chips_w10 = merged[
        (merged["stock_sku"] == "chips") & (merged["week_label"] == "2026-W10")
    ].iloc[0]
    assert chips_w10["usage"] == 99.0
    assert chips_w10["sales_count"] == 99

    # Les autres lignes restent intactes
    assert len(merged) == 5
    coca_w12 = merged[
        (merged["stock_sku"] == "coca") & (merged["week_label"] == "2026-W12")
    ].iloc[0]
    assert coca_w12["usage"] == 3.0


def test_update_appends_new_weeks(tmp_parquet, sample_weekly_usage, sample_sales_count):
    ds.update_weekly_usage(sample_weekly_usage, sample_sales_count, tmp_parquet)

    new_week = {"chips": {"2026-W13": 5.0}, "coca": {"2026-W13": 2.0}}
    new_sales = {"chips": {"2026-W13": 5}, "coca": {"2026-W13": 2}}
    merged = ds.update_weekly_usage(new_week, new_sales, tmp_parquet)

    assert len(merged) == 7
    w13 = merged[merged["week_label"] == "2026-W13"]
    assert len(w13) == 2


def test_merge_with_empty_existing(sample_weekly_usage, sample_sales_count):
    new_df = ds.weekly_usage_dict_to_dataframe(sample_weekly_usage, sample_sales_count)
    merged = ds.merge_weekly_usage(ds._empty_dataframe(), new_df)
    assert len(merged) == 5


def test_merge_with_empty_new(sample_weekly_usage, sample_sales_count):
    existing = ds.weekly_usage_dict_to_dataframe(sample_weekly_usage, sample_sales_count)
    merged = ds.merge_weekly_usage(existing, ds._empty_dataframe())
    assert len(merged) == 5


def test_filter_skus(sample_weekly_usage, sample_sales_count):
    df = ds.weekly_usage_dict_to_dataframe(sample_weekly_usage, sample_sales_count)
    filtered = ds.filter_skus(df, ["chips"])
    assert set(filtered["stock_sku"].unique()) == {"chips"}
    assert len(filtered) == 2


def test_dict_to_dataframe_empty():
    df = ds.weekly_usage_dict_to_dataframe({}, {})
    assert len(df) == 0
    assert list(df.columns) == ds.SCHEMA_COLUMNS


def test_sorted_after_save(tmp_parquet):
    unsorted = {
        "z_sku": {"2026-W12": 1.0, "2026-W10": 2.0},
        "a_sku": {"2026-W11": 3.0},
    }
    ds.update_weekly_usage(unsorted, None, tmp_parquet)
    reloaded = ds.load_weekly_usage(tmp_parquet)
    assert reloaded["stock_sku"].tolist() == ["a_sku", "z_sku", "z_sku"]
    z_rows = reloaded[reloaded["stock_sku"] == "z_sku"]
    assert z_rows["week"].tolist() == [10, 12]
