"""Tests de l'agrégation de consommation, dont le multi-stock (also_consumes).

Couvre le cas « un Kir consomme du vin blanc ET de la crème de cassis/mûre » :
une même vente décrémente plusieurs stock_sku, chacun filtré par sa propre ancre.
"""

from datetime import date

from stocks.sumup_stocks import (
    aggregate_stock_usage_since,
    build_sku_index,
    compute_excel_purchase_coverage,
)


def _txn(name: str, qty: int, day: str, variant: str = ""):
    return {
        "status": "SUCCESSFUL",
        "timestamp": f"{day}T12:00:00Z",
        "products": [{"name": name, "description": variant, "quantity": qty}],
    }


ANCHOR = date(2026, 1, 1)
AS_OF = date(2026, 6, 1)


def _kir_item(creme_per_sale=0.02):
    return {
        "stock_sku": "vin_blanc",
        "consumption_per_sale": 0.15,
        "sumup_match": {"name": "Kir", "variant": ""},
        "also_consumes": [
            {"stock_sku": "creme_cassis_mure", "consumption_per_sale": creme_per_sale}
        ],
    }


class TestAlsoConsumes:
    def test_kir_consumes_wine_and_creme(self):
        idx = build_sku_index([_kir_item()])
        anchors = {"vin_blanc": ANCHOR, "creme_cassis_mure": ANCHOR}
        usage = aggregate_stock_usage_since([_txn("Kir", 2, "2026-03-10")], idx, anchors, AS_OF)
        assert usage["vin_blanc"] == 2 * 0.15
        assert usage["creme_cassis_mure"] == 2 * 0.02

    def test_secondary_gated_by_its_own_anchor(self):
        """Si la crème n'a pas d'ancre, sa consommation n'est pas comptée."""
        idx = build_sku_index([_kir_item()])
        anchors = {"vin_blanc": ANCHOR}  # pas d'ancre crème
        usage = aggregate_stock_usage_since([_txn("Kir", 1, "2026-03-10")], idx, anchors, AS_OF)
        assert usage["vin_blanc"] == 0.15
        assert "creme_cassis_mure" not in usage

    def test_secondary_anchor_excludes_old_sales(self):
        """Une vente antérieure à l'ancre crème ne décrémente pas la crème."""
        idx = build_sku_index([_kir_item()])
        anchors = {"vin_blanc": date(2026, 1, 1), "creme_cassis_mure": date(2026, 4, 1)}
        usage = aggregate_stock_usage_since([_txn("Kir", 1, "2026-03-10")], idx, anchors, AS_OF)
        assert usage["vin_blanc"] == 0.15           # après l'ancre vin
        assert "creme_cassis_mure" not in usage     # avant l'ancre crème

    def test_item_without_also_consumes_unchanged(self):
        idx = build_sku_index([{
            "stock_sku": "chips",
            "consumption_per_sale": 1,
            "sumup_match": {"name": "Chips", "variant": ""},
        }])
        anchors = {"chips": ANCHOR}
        usage = aggregate_stock_usage_since([_txn("Chips", 3, "2026-03-10")], idx, anchors, AS_OF)
        assert usage["chips"] == 3
        assert len(usage) == 1

    def test_zero_consumption_ignored(self):
        idx = build_sku_index([_kir_item(creme_per_sale=0)])
        anchors = {"vin_blanc": ANCHOR, "creme_cassis_mure": ANCHOR}
        usage = aggregate_stock_usage_since([_txn("Kir", 5, "2026-03-10")], idx, anchors, AS_OF)
        assert usage["vin_blanc"] == 5 * 0.15
        assert "creme_cassis_mure" not in usage


class TestExcelPurchaseCoverage:
    """compute_excel_purchase_coverage : alerte de correspondance Excel ↔ stock."""

    def _kpis(self, *skus):
        return [{"stock_sku": s, "label": s.title()} for s in skus]

    def _map(self, *skus):
        return [{"excel_label": s, "stock_sku": s} for s in skus]

    def test_article_without_excel_line_is_unlinked(self):
        kpis = self._kpis("vin_blanc", "cidre")
        mapping = self._map("vin_blanc")  # cidre absent du mapping
        unlinked, orphans = compute_excel_purchase_coverage(kpis, mapping)
        assert [k["stock_sku"] for k in unlinked] == ["cidre"]
        assert orphans == []

    def test_mapping_to_inactive_article_is_orphan(self):
        kpis = self._kpis("vin_blanc")
        mapping = self._map("vin_blanc", "produit_disparu")
        unlinked, orphans = compute_excel_purchase_coverage(kpis, mapping)
        assert unlinked == []
        assert [e["stock_sku"] for e in orphans] == ["produit_disparu"]

    def test_full_coverage_no_alert(self):
        kpis = self._kpis("vin_blanc", "cidre")
        mapping = self._map("vin_blanc", "cidre")
        unlinked, orphans = compute_excel_purchase_coverage(kpis, mapping)
        assert unlinked == []
        assert orphans == []

    def test_empty_mapping_flags_all(self):
        kpis = self._kpis("vin_blanc", "cidre")
        unlinked, orphans = compute_excel_purchase_coverage(kpis, [])
        assert {k["stock_sku"] for k in unlinked} == {"vin_blanc", "cidre"}
        assert orphans == []
