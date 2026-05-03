"""Tests du feature engineering ML."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocks.ml import dataset as ds
from stocks.ml import features as ft


@pytest.fixture
def synthetic_history():
    """20 semaines d'historique pour 2 SKU avec saisonnalité simple."""
    rows = []
    base_dates = pd.date_range("2025-12-01", periods=20, freq="W-MON")
    for sku, amplitude in [("chips", 10.0), ("coca", 5.0)]:
        for i, ts in enumerate(base_dates):
            iso = ts.isocalendar()
            rows.append(
                {
                    "stock_sku": sku,
                    "week_label": f"{iso.year}-W{iso.week:02d}",
                    "year": iso.year,
                    "week": iso.week,
                    "week_start": ts.date(),
                    "usage": amplitude + i * 0.5,
                    "sales_count": int(amplitude),
                }
            )
    return ds._coerce_dtypes(pd.DataFrame(rows))


def test_easter_calculation():
    assert ft._easter_sunday(2026) == date(2026, 4, 5)
    assert ft._easter_sunday(2024) == date(2024, 3, 31)
    assert ft._easter_sunday(2025) == date(2025, 4, 20)


def test_fr_public_holidays_count():
    holidays_2026 = ft.fr_public_holidays(2026)
    assert len(holidays_2026) == 11
    assert date(2026, 1, 1) in holidays_2026
    assert date(2026, 12, 25) in holidays_2026
    assert date(2026, 5, 1) in holidays_2026


def test_holidays_in_week():
    # Semaine du 1er mai 2026 (vendredi)
    assert ft._holidays_in_week(date(2026, 4, 27)) == 1
    # Semaine sans férié
    assert ft._holidays_in_week(date(2026, 3, 9)) == 0
    # Semaine du 14 juillet 2026 (mardi)
    assert ft._holidays_in_week(date(2026, 7, 13)) == 1


def test_add_calendar_features(synthetic_history):
    df = ft.add_calendar_features(synthetic_history)
    assert "month" in df.columns
    assert "week_of_year" in df.columns
    assert "n_holidays" in df.columns
    assert "sin_week" in df.columns
    assert "cos_week" in df.columns
    assert (df["month"].between(1, 12)).all()
    assert (df["week_in_month"].between(1, 5)).all()
    # Encodage cyclique : sin² + cos² ≈ 1
    assert np.allclose(df["sin_week"]**2 + df["cos_week"]**2, 1.0)


def test_add_lag_features_no_leakage(synthetic_history):
    df = ft.add_lag_features(synthetic_history, lags=(1, 2, 4))
    chips = df[df["stock_sku"] == "chips"].sort_values(["year", "week"]).reset_index(drop=True)
    # La 1ère ligne n'a pas de lag_1
    assert pd.isna(chips.loc[0, "lag_1"])
    # La 2ème ligne a lag_1 = usage de la 1ère
    assert chips.loc[1, "lag_1"] == chips.loc[0, "usage"]
    # La 5ème ligne a lag_4 = usage de la 1ère
    assert chips.loc[4, "lag_4"] == chips.loc[0, "usage"]


def test_lags_isolated_per_sku(synthetic_history):
    df = ft.add_lag_features(synthetic_history, lags=(1,))
    # La 1ère ligne de chaque SKU n'a pas de lag_1 (pas de croisement entre SKU)
    chips = df[df["stock_sku"] == "chips"].sort_values(["year", "week"]).reset_index(drop=True)
    coca = df[df["stock_sku"] == "coca"].sort_values(["year", "week"]).reset_index(drop=True)
    assert pd.isna(chips.loc[0, "lag_1"])
    assert pd.isna(coca.loc[0, "lag_1"])


def test_rolling_features_shift_avoids_leakage(synthetic_history):
    df = ft.add_rolling_features(synthetic_history, windows=(4,))
    chips = df[df["stock_sku"] == "chips"].sort_values(["year", "week"]).reset_index(drop=True)
    # La rolling_mean_4 à la ligne i ne doit PAS inclure la valeur de la ligne i
    for i in range(4, len(chips)):
        prev_4 = chips.loc[i - 4 : i - 1, "usage"].mean()
        assert chips.loc[i, "rolling_mean_4"] == pytest.approx(prev_4)


def test_prepare_training_table_drops_warmup(synthetic_history):
    X, y, meta = ft.prepare_training_table(synthetic_history, lags=(1, 2, 4), windows=(4,))
    # Avec lags max=4 et 20 semaines par SKU, on garde 16 par SKU = 32 lignes
    assert len(X) == 32
    assert len(y) == 32
    assert len(meta) == 32
    assert not X[[f"lag_{k}" for k in (1, 2, 4)]].isna().any().any()


def test_prepare_training_table_columns(synthetic_history):
    X, _, _ = ft.prepare_training_table(synthetic_history, lags=(1, 2), windows=(4,))
    expected = {
        "stock_sku", "month", "week_of_year", "week_in_month", "n_holidays",
        "is_first_week_of_month", "is_last_week_of_month",
        "sin_week", "cos_week",
        "lag_1", "lag_2",
        "rolling_mean_4", "rolling_std_4",
    }
    assert set(X.columns) == expected


def test_prepare_training_meta_traceable(synthetic_history):
    X, y, meta = ft.prepare_training_table(synthetic_history, lags=(1,), windows=(4,))
    assert len(meta) == len(X) == len(y)
    assert "stock_sku" in meta.columns
    assert "week_label" in meta.columns


def test_build_feature_table_keeps_all_rows(synthetic_history):
    df = ft.build_feature_table(synthetic_history, lags=(1, 2, 4), windows=(4,))
    assert len(df) == len(synthetic_history)
