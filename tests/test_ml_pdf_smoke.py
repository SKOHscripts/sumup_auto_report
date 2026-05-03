"""Smoke test : la génération PDF doit fonctionner avec ou sans ML attaché.

But : vérifier que les hooks d'intégration PDF (helpers _ml_rupture_label,
_ml_rupture_range, et le rendu de la bande de confiance) ne plantent pas
quand un kpi a (ou n'a pas) la clé ml_projection.
"""
import os
from datetime import date, timedelta

os.environ.setdefault("SUMUP_API_KEY", "test_placeholder")

from stocks import sumup_stocks  # noqa: E402


def _fake_kpi(with_ml: bool) -> dict:
    base = {
        "stock_sku": "chips",
        "label": "Chips",
        "category": "snacking",
        "unit": "piece",
        "sumup_match": {},
        "linked_items": [],
        "linked_items_count": 0,
        "stock_on_hand": 50.0,
        "stock_reserved": 0.0,
        "available_stock": 50.0,
        "incoming_qty": 0.0,
        "incoming_eta": None,
        "last_inventory_date": "2026-04-01",
        "inventory_method": "manual",
        "weekly_demand": 10.0,
        "lead_time_weeks": 2,
        "safety_stock": 20.0,
        "reorder_point": 30.0,
        "target_stock": 60.0,
        "sales_series": [10, 12, 8, 11],
        "usage_series": [10, 12, 8, 11],
        "sales_count_series": [10, 12, 8, 11],
        "weeks_range": ["2026-W14", "2026-W15", "2026-W16", "2026-W17"],
        "total_sold": 41,
        "total_used": 41,
        "sales_7d": 11,
        "usage_7d": 11,
        "sales_28d": 41,
        "usage_28d": 41,
        "avg_weekly": 10.25,
        "avg_rolling4": 10.25,
        "variation_pct": 5.0,
        "n_zero_weeks": 0,
        "proj_next_week": 10.0,
        "proj_4_weeks": 41.0,
        "coverage_weeks": 4.9,
        "rupture_date": "2026-06-08",
        "qty_to_order": 10.0,
        "status": "OK",
    }
    if with_ml:
        today = date(2026, 5, 4)
        base["ml_projection"] = {
            "rupture_date_p10": "2026-05-29",
            "rupture_date_p50": "2026-06-12",
            "rupture_date_p90": "2026-07-02",
            "prob_rupture": 0.92,
            "weekly_forecast": [
                {
                    "week_start": (today + timedelta(weeks=i)).isoformat(),
                    "q_low": 9.0, "q_med": 10.0, "q_high": 11.5,
                }
                for i in range(8)
            ],
            "model_version": "abc123",
            "model_trained_at": "2026-05-02T22:00:00+00:00",
        }
    return base


def test_ml_helpers_with_projection():
    kpi = _fake_kpi(with_ml=True)
    assert sumup_stocks._ml_rupture_label(kpi) == "12/06/2026"
    assert sumup_stocks._ml_rupture_range(kpi) == "29/05/2026 -> 02/07/2026"


def test_ml_helpers_without_projection():
    kpi = _fake_kpi(with_ml=False)
    assert sumup_stocks._ml_rupture_label(kpi) == "N/A"
    assert sumup_stocks._ml_rupture_range(kpi) == "N/A"


def test_pdf_generation_with_ml(tmp_path):
    """generate_pdf doit fonctionner quand un kpi contient ml_projection."""
    kpi = _fake_kpi(with_ml=True)
    out = tmp_path / "rapport.pdf"
    sumup_stocks.generate_pdf([kpi], unmapped=[], week_label="2026-W18",
                              weeks_range=["2026-W14", "2026-W15", "2026-W16", "2026-W17"],
                              path=str(out))
    assert out.exists()
    assert out.stat().st_size > 1000


def test_pdf_generation_without_ml(tmp_path):
    """generate_pdf doit aussi fonctionner sans ml_projection (régression)."""
    kpi = _fake_kpi(with_ml=False)
    out = tmp_path / "rapport.pdf"
    sumup_stocks.generate_pdf([kpi], unmapped=[], week_label="2026-W18",
                              weeks_range=["2026-W14", "2026-W15", "2026-W16", "2026-W17"],
                              path=str(out))
    assert out.exists()
    assert out.stat().st_size > 1000
