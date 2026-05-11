"""Tests unitaires pour stocks.ml.anomaly — détection d'anomalies."""
import pandas as pd

from stocks.ml.anomaly import (
    DEFAULT_Z_THRESHOLD,
    MIN_WEEKS_FOR_ZSCORE,
    anomaly_summary,
    detect_anomalies,
)


def _make_history(usages: list[float], sku: str = "SKU_A") -> pd.DataFrame:
    rows = []
    for i, u in enumerate(usages):
        year, week = 2026, i + 1
        rows.append({
            "stock_sku": sku,
            "week_label": f"{year}-W{week:02d}",
            "year": year,
            "week": week,
            "usage": float(u),
        })
    return pd.DataFrame(rows)


class TestDetectAnomalies:
    def test_returns_copy_with_new_columns(self):
        df = _make_history([10.0] * 10)
        result = detect_anomalies(df)
        assert "z_score" in result.columns
        assert "is_anomaly" in result.columns
        # Original not mutated
        assert "z_score" not in df.columns

    def test_obvious_spike_flagged(self):
        # 9 normal values + 1 massive spike
        usages = [10.0] * 9 + [1000.0]
        df = _make_history(usages)
        result = detect_anomalies(df)
        assert result.iloc[-1]["is_anomaly"] is True or result.iloc[-1]["is_anomaly"] == True

    def test_stable_series_no_anomaly(self):
        usages = [10.0] * 20
        df = _make_history(usages)
        result = detect_anomalies(df)
        # std=0 → skipped, no anomaly flagged
        assert result["is_anomaly"].sum() == 0

    def test_too_few_weeks_no_zscore(self):
        df = _make_history([5.0] * (MIN_WEEKS_FOR_ZSCORE - 1))
        result = detect_anomalies(df)
        assert result["z_score"].isna().all()
        assert result["is_anomaly"].sum() == 0

    def test_multiple_skus_independent(self):
        df_a = _make_history([10.0] * 9 + [500.0], sku="SKU_A")
        df_b = _make_history([100.0] * 10, sku="SKU_B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        result = detect_anomalies(df)
        # Spike only in SKU_A
        anomalies_a = result[result["stock_sku"] == "SKU_A"]["is_anomaly"].sum()
        anomalies_b = result[result["stock_sku"] == "SKU_B"]["is_anomaly"].sum()
        assert anomalies_a >= 1
        assert anomalies_b == 0

    def test_custom_threshold(self):
        usages = [10.0] * 9 + [30.0]  # mild spike
        df = _make_history(usages)
        # Very tight threshold — should flag the spike
        result_tight = detect_anomalies(df, k=1.0)
        # Loose threshold — should not flag it
        result_loose = detect_anomalies(df, k=5.0)
        assert result_tight["is_anomaly"].sum() >= 1
        assert result_loose["is_anomaly"].sum() == 0

    def test_output_has_same_length(self):
        df = _make_history([10.0] * 15)
        result = detect_anomalies(df)
        assert len(result) == len(df)

    def test_default_threshold_constant(self):
        assert DEFAULT_Z_THRESHOLD == 2.5

    def test_min_weeks_constant(self):
        assert MIN_WEEKS_FOR_ZSCORE >= 3


class TestAnomalySummary:
    def test_returns_dataframe(self):
        df = _make_history([10.0] * 9 + [500.0])
        flagged = detect_anomalies(df)
        summary = anomaly_summary(flagged)
        assert isinstance(summary, pd.DataFrame)
        assert "stock_sku" in summary.columns
        assert "n_anomalies" in summary.columns
        assert "pct_anomalies" in summary.columns
        assert "max_z" in summary.columns

    def test_summary_counts_match(self):
        # Use a low threshold so both spikes are flagged
        usages = [10.0] * 8 + [500.0, 600.0]
        df = _make_history(usages)
        flagged = detect_anomalies(df, k=1.0)
        summary = anomaly_summary(flagged)
        assert len(summary) == 1  # 1 SKU
        assert summary.iloc[0]["n_anomalies"] >= 1

    def test_summary_sorted_by_n_anomalies_desc(self):
        df_a = _make_history([10.0] * 8 + [500.0, 600.0], sku="SKU_A")
        df_b = _make_history([10.0] * 9 + [500.0], sku="SKU_B")
        df = pd.concat([df_a, df_b], ignore_index=True)
        flagged = detect_anomalies(df, k=2.0)
        summary = anomaly_summary(flagged)
        assert summary.iloc[0]["n_anomalies"] >= summary.iloc[1]["n_anomalies"]

    def test_pct_anomalies_range(self):
        df = _make_history([10.0] * 9 + [500.0])
        flagged = detect_anomalies(df)
        summary = anomaly_summary(flagged)
        pct = float(summary.iloc[0]["pct_anomalies"])
        assert 0.0 <= pct <= 1.0
