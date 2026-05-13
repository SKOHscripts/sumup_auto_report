"""Tests unitaires pour stocks/sumup_statistics.py."""
import json
import os

import pytest

# Valeur fictive requise avant l'import du module (load_project_env au niveau module)
os.environ.setdefault("SUMUP_API_KEY", "test_placeholder")


# ── Fixtures locales ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def Catalog():
    from stocks.sumup_statistics import Catalog as _Catalog
    return _Catalog


@pytest.fixture(scope="module")
def CatalogItem():
    from stocks.sumup_statistics import CatalogItem as _CatalogItem
    return _CatalogItem


@pytest.fixture(scope="module")
def SumUpClient():
    from stocks.sumup_statistics import SumUpClient as _SumUpClient
    return _SumUpClient


@pytest.fixture(scope="module")
def TransactionAnalyzer():
    from stocks.sumup_statistics import TransactionAnalyzer as _TransactionAnalyzer
    return _TransactionAnalyzer


@pytest.fixture
def catalog_items():
    return [
        {
            "stock_sku": "chips",
            "label": "Chips",
            "stock_label": "Chips",
            "category": "snacking",
            "unit": "piece",
            "enabled": True,
            "is_stock_reference": True,
            "consumption_per_sale": 1,
            "sumup_match": {"name": "Chips", "variant": ""},
            "stock_state": {"stock_on_hand": 10},
        },
        {
            "stock_sku": "coca",
            "label": "Coca-Cola",
            "stock_label": "Coca-Cola",
            "category": "soft",
            "unit": "piece",
            "enabled": True,
            "is_stock_reference": True,
            "consumption_per_sale": 1,
            "sumup_match": {"name": "Jus & Sodas", "variant": "Coca Cola classique"},
            "stock_state": {"stock_on_hand": 24},
        },
        {
            "stock_sku": "disabled_item",
            "label": "Désactivé",
            "category": "autres",
            "unit": "piece",
            "enabled": False,
            "is_stock_reference": True,
            "consumption_per_sale": 1,
            "sumup_match": {"name": "Disabled", "variant": ""},
        },
    ]


# ── CatalogItem ───────────────────────────────────────────────────────────────

class TestCatalogItem:
    """Tests de CatalogItem.display_name."""

    def _make(self, CatalogItem, label="", stocksku="sku_test"):
        return CatalogItem(
            stocksku=stocksku,
            label=label,
            category="test",
            unit="piece",
            enabled=True,
            is_reference=True,
            sumup_name="",
            sumup_variant="",
            consumption_per_sale=1.0,
            sale_price=None,
            raw={},
        )

    def test_display_name_returns_label_when_set(self, CatalogItem):
        item = self._make(CatalogItem, label="Chips")
        assert item.display_name == "Chips"

    def test_display_name_falls_back_to_sku(self, CatalogItem):
        item = self._make(CatalogItem, label="", stocksku="chips_sku")
        assert item.display_name == "chips_sku"


# ── Catalog ───────────────────────────────────────────────────────────────────

class TestCatalog:
    """Tests de la classe Catalog."""

    def test_disabled_items_excluded(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        labels = [i.label for i in cat.items]
        assert "Désactivé" not in labels

    def test_enabled_items_included(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        assert len(cat.items) == 2

    def test_match_product_exact_name(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        item = cat.match_product("Chips", "")
        assert item is not None
        assert item.stocksku == "chips"

    def test_match_product_with_variant(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        item = cat.match_product("Jus & Sodas", "Coca Cola classique")
        assert item is not None
        assert item.stocksku == "coca"

    def test_match_product_unknown_returns_none(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        assert cat.match_product("Produit Inconnu XYZ", "") is None

    def test_match_product_partial_name_fallback(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        item = cat.match_product("Chips maison", "")
        assert item is not None

    def test_match_product_ignores_variant_when_empty(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        item = cat.match_product("Chips", "variante_inconnue")
        assert item is not None
        assert item.stocksku == "chips"

    def test_sku_index_built_correctly(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        assert len(cat.sku_index) == 2

    def test_reference_by_sku_contains_enabled_skus(self, Catalog, catalog_items):
        cat = Catalog(catalog_items)
        assert "chips" in cat.reference_by_sku
        assert "coca" in cat.reference_by_sku
        assert "disabled_item" not in cat.reference_by_sku

    def test_from_path_loads_correctly(self, Catalog, catalog_items, tmp_path):
        p = tmp_path / "items.json"
        p.write_text(json.dumps(catalog_items), encoding="utf-8")
        cat = Catalog.from_path(p)
        assert len(cat.items) == 2

    def test_from_path_raises_on_non_list(self, Catalog, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="liste"):
            Catalog.from_path(p)

    def test_empty_catalog_returns_no_items(self, Catalog):
        cat = Catalog([])
        assert cat.items == []


# ── SumUpClient ───────────────────────────────────────────────────────────────

class TestSumUpClientFetchTransactions:
    """Tests de SumUpClient.fetch_transactions."""

    def test_fetch_from_mock_file_as_list(self, SumUpClient, tmp_path):
        txns = [{"id": "t1", "amount": 10.0}]
        f = tmp_path / "mock.json"
        f.write_text(json.dumps(txns), encoding="utf-8")
        client = SumUpClient(api_key=None)
        result = client.fetch_transactions("2026-01-01", "2026-04-30", mock_file=str(f))
        assert result == txns

    def test_fetch_from_mock_file_dict_items(self, SumUpClient, tmp_path):
        txns = [{"id": "t1"}, {"id": "t2"}]
        f = tmp_path / "mock.json"
        f.write_text(json.dumps({"items": txns}), encoding="utf-8")
        client = SumUpClient(api_key=None)
        result = client.fetch_transactions("2026-01-01", "2026-04-30", mock_file=str(f))
        assert len(result) == 2

    def test_fetch_from_mock_file_dict_transactions_key(self, SumUpClient, tmp_path):
        txns = [{"id": "t3"}]
        f = tmp_path / "mock.json"
        f.write_text(json.dumps({"transactions": txns}), encoding="utf-8")
        client = SumUpClient(api_key=None)
        result = client.fetch_transactions("2026-01-01", "2026-04-30", mock_file=str(f))
        assert result == txns

    def test_no_api_key_no_mock_raises_runtime_error(self, SumUpClient):
        client = SumUpClient(api_key=None)
        with pytest.raises(RuntimeError, match="SUMUP_API_KEY"):
            client.fetch_transactions("2026-01-01", "2026-04-30")

    def test_authorization_header_set_with_api_key(self, SumUpClient):
        client = SumUpClient(api_key="my_test_key")
        assert "Authorization" in client.headers
        assert "my_test_key" in client.headers["Authorization"]

    def test_no_api_key_no_headers(self, SumUpClient):
        client = SumUpClient(api_key=None)
        assert client.headers == {}


# ── TransactionAnalyzer.extract_products ─────────────────────────────────────

class TestTransactionAnalyzerExtractProducts:
    """Tests de TransactionAnalyzer.extract_products."""

    @pytest.fixture
    def analyzer(self, TransactionAnalyzer, Catalog):
        return TransactionAnalyzer(Catalog([]))

    def test_list_of_products_returned(self, analyzer):
        txn = {"products": [
            {"name": "Chips", "description": "", "quantity": 2, "price": 1.5, "total_price": 3.0},
        ]}
        products = analyzer.extract_products(txn)
        assert len(products) == 1
        assert products[0]["name"] == "Chips"
        assert products[0]["quantity"] == 2

    def test_empty_products_uses_summary_field(self, analyzer):
        txn = {"products": [], "product_summary": "Café", "amount": 2.0}
        products = analyzer.extract_products(txn)
        assert len(products) == 1
        assert products[0]["name"] == "Café"

    def test_no_products_no_summary_returns_empty(self, analyzer):
        assert analyzer.extract_products({"products": []}) == []

    def test_quantity_defaults_to_one(self, analyzer):
        txn = {"products": [{"name": "Coca", "description": ""}]}
        products = analyzer.extract_products(txn)
        assert products[0]["quantity"] == 1

    def test_multiple_products(self, analyzer):
        txn = {"products": [
            {"name": "A", "description": "", "quantity": 1},
            {"name": "B", "description": "", "quantity": 3},
        ]}
        products = analyzer.extract_products(txn)
        assert len(products) == 2

    def test_non_dict_product_skipped(self, analyzer):
        txn = {"products": ["invalid_entry", {"name": "Valid", "description": ""}]}
        products = analyzer.extract_products(txn)
        assert len(products) == 1
        assert products[0]["name"] == "Valid"


# ── TransactionAnalyzer.detect_payment_method ────────────────────────────────

class TestTransactionAnalyzerDetectPaymentMethod:
    """Tests de TransactionAnalyzer.detect_payment_method."""

    @pytest.fixture
    def analyzer(self, TransactionAnalyzer, Catalog):
        return TransactionAnalyzer(Catalog([]))

    def test_cash_payment_type(self, analyzer):
        assert analyzer.detect_payment_method({"payment_type": "cash"}) == "cash"

    def test_card_payment_returns_cb(self, analyzer):
        assert analyzer.detect_payment_method({"payment_type": "card"}) == "cb"

    def test_especes_detected_as_cash(self, analyzer):
        assert analyzer.detect_payment_method({"payment_method": "especes"}) == "cash"

    def test_empty_transaction_defaults_to_cb(self, analyzer):
        assert analyzer.detect_payment_method({}) == "cb"

    def test_none_values_handled(self, analyzer):
        txn = {"payment_type": None, "payment_method": None}
        result = analyzer.detect_payment_method(txn)
        assert result in ("cash", "cb")


# ── TransactionAnalyzer.normalize_transactions ────────────────────────────────

class TestTransactionAnalyzerNormalizeTransactions:
    """Tests de TransactionAnalyzer.normalize_transactions."""

    @pytest.fixture
    def analyzer(self, TransactionAnalyzer, Catalog, catalog_items):
        return TransactionAnalyzer(Catalog(catalog_items))

    def _make_txn(self, product_name="Chips", status="SUCCESSFUL", qty=1):
        return {
            "id": "t1",
            "status": status,
            "timestamp": "2026-04-20T10:00:00",
            "amount": 1.5,
            "products": [{"name": product_name, "description": "", "quantity": qty, "price": 1.5}],
        }

    def test_failed_transactions_excluded(self, analyzer):
        rows = analyzer.normalize_transactions([self._make_txn(status="FAILED")])
        assert rows == []

    def test_cancelled_transactions_excluded(self, analyzer):
        rows = analyzer.normalize_transactions([self._make_txn(status="CANCELLED")])
        assert rows == []

    def test_valid_transaction_produces_row(self, analyzer):
        rows = analyzer.normalize_transactions([self._make_txn(qty=2)])
        assert len(rows) == 1
        assert rows[0]["product_name"] == "Chips"
        assert rows[0]["quantity"] == 2

    def test_missing_timestamp_excluded(self, analyzer):
        txn = {"id": "t2", "status": "SUCCESSFUL",
               "products": [{"name": "Chips", "description": "", "quantity": 1}]}
        assert analyzer.normalize_transactions([txn]) == []

    def test_mapped_flag_true_for_known_product(self, analyzer):
        rows = analyzer.normalize_transactions([self._make_txn("Chips")])
        assert rows[0]["mapped"] is True

    def test_mapped_flag_false_for_unknown_product(self, analyzer):
        rows = analyzer.normalize_transactions([self._make_txn("ProduitInconnu")])
        assert rows[0]["mapped"] is False

    def test_week_label_set(self, analyzer):
        rows = analyzer.normalize_transactions([self._make_txn()])
        assert "week" in rows[0]
        assert "-W" in rows[0]["week"]

    def test_empty_input_returns_empty(self, analyzer):
        assert analyzer.normalize_transactions([]) == []

    def test_no_products_row_skipped(self, analyzer):
        txn = {"id": "t3", "status": "SUCCESSFUL", "timestamp": "2026-04-20T10:00:00",
               "products": []}
        assert analyzer.normalize_transactions([txn]) == []


# ── TransactionAnalyzer.compute_metrics ──────────────────────────────────────

class TestTransactionAnalyzerComputeMetrics:
    """Tests de TransactionAnalyzer.compute_metrics."""

    @pytest.fixture
    def analyzer(self, TransactionAnalyzer, Catalog):
        return TransactionAnalyzer(Catalog([]))

    def _row(self, label="Chips", qty=1, category="snacking", payment="cb",
             revenue=1.5, mapped=True, week="2026-W16"):
        return {
            "label": label,
            "quantity": qty,
            "category": category,
            "payment_method": payment,
            "estimated_revenue": revenue,
            "mapped": mapped,
            "week": week,
            "product_name": label,
            "product_variant": "",
        }

    def test_total_qty_summed(self, analyzer):
        rows = [self._row(qty=3), self._row(qty=5)]
        assert analyzer.compute_metrics(rows)["total_qty"] == 8

    def test_total_revenue_summed(self, analyzer):
        rows = [self._row(revenue=10.0), self._row(revenue=5.5)]
        metrics = analyzer.compute_metrics(rows)
        assert abs(metrics["total_revenue"] - 15.5) < 0.01

    def test_payment_counts_aggregated(self, analyzer):
        rows = [self._row(payment="cash"), self._row(payment="cb"), self._row(payment="cash")]
        counts = analyzer.compute_metrics(rows)["payment_counts"]
        assert counts["cash"] == 2
        assert counts["cb"] == 1

    def test_unmapped_products_listed(self, analyzer):
        rows = [self._row(mapped=False, label="Inconnu")]
        metrics = analyzer.compute_metrics(rows)
        assert len(metrics["unmapped"]) == 1
        assert metrics["unmapped"][0]["name"] == "Inconnu"

    def test_mapped_products_not_in_unmapped(self, analyzer):
        rows = [self._row(mapped=True, label="Chips")]
        metrics = analyzer.compute_metrics(rows)
        assert metrics["unmapped"] == []

    def test_top_articles_sorted_by_qty_descending(self, analyzer):
        rows = [self._row(label="A", qty=5), self._row(label="B", qty=10), self._row(label="C", qty=1)]
        top = analyzer.compute_metrics(rows)["top_articles"]
        assert top[0]["label"] == "B"

    def test_by_category_aggregated(self, analyzer):
        rows = [self._row(category="bar", qty=2), self._row(category="bar", qty=3),
                self._row(category="soft", qty=1)]
        by_cat = analyzer.compute_metrics(rows)["by_category"]
        assert by_cat["bar"]["qty"] == 5
        assert by_cat["soft"]["qty"] == 1

    def test_weeks_sorted(self, analyzer):
        rows = [self._row(week="2026-W18"), self._row(week="2026-W16"), self._row(week="2026-W17")]
        weeks = analyzer.compute_metrics(rows)["weeks"]
        assert weeks == sorted(weeks)

    def test_empty_rows_returns_zero_metrics(self, analyzer):
        metrics = analyzer.compute_metrics([])
        assert metrics["total_qty"] == 0
        assert metrics["total_revenue"] == 0.0
        assert metrics["weeks"] == []

    def test_mapped_rows_count(self, analyzer):
        rows = [self._row(mapped=True), self._row(mapped=False), self._row(mapped=True)]
        metrics = analyzer.compute_metrics(rows)
        assert metrics["mapped_rows"] == 2
        assert metrics["total_rows"] == 3

    def test_least_articles_sorted_by_qty_ascending(self, analyzer):
        rows = [self._row(label="A", qty=10), self._row(label="B", qty=1)]
        least = analyzer.compute_metrics(rows)["least_articles"]
        assert least[0]["label"] == "B"

    # ── Normalisation ─────────────────────────────────────────────────────────

    def test_qty_per_week_computed_single_week(self, analyzer):
        rows = [self._row(label="A", qty=8, week="2026-W16")]
        art = analyzer.compute_metrics(rows)["top_articles"][0]
        assert art["qty_per_week"] == 8.0
        assert art["n_active_weeks"] == 1

    def test_qty_per_week_computed_two_weeks(self, analyzer):
        rows = [
            self._row(label="A", qty=4, week="2026-W16"),
            self._row(label="A", qty=4, week="2026-W17"),
        ]
        art = analyzer.compute_metrics(rows)["top_articles"][0]
        assert art["n_active_weeks"] == 2
        assert art["qty_per_week"] == 4.0  # 8 / 2

    def test_norm_weeks_fixed_overrides_active_weeks(self, analyzer):
        rows = [
            self._row(label="A", qty=4, week="2026-W16"),
            self._row(label="A", qty=4, week="2026-W17"),
        ]
        art = analyzer.compute_metrics(rows, norm_weeks=8)["top_articles"][0]
        assert art["qty_per_week"] == 1.0  # 8 / 8

    def test_normalization_changes_ranking(self, analyzer):
        # B has more total qty but spread over more weeks → lower rate
        # A has fewer total qty but in fewer weeks → higher rate
        rows = [
            self._row(label="A", qty=6, week="2026-W16"),
            self._row(label="B", qty=3, week="2026-W16"),
            self._row(label="B", qty=3, week="2026-W17"),
            self._row(label="B", qty=3, week="2026-W18"),
        ]
        # Without normalization B would have qty=9 > A's qty=6
        # With normalization: A=6/1=6/sem, B=9/3=3/sem → A ranks first
        top = analyzer.compute_metrics(rows)["top_articles"]
        assert top[0]["label"] == "A"

    def test_norm_label_auto(self, analyzer):
        metrics = analyzer.compute_metrics([self._row()])
        assert metrics["norm_label"] == "sem. actives par article"

    def test_norm_label_fixed(self, analyzer):
        metrics = analyzer.compute_metrics([self._row()], norm_weeks=4)
        assert "4" in metrics["norm_label"]

    def test_norm_weeks_in_metrics(self, analyzer):
        metrics = analyzer.compute_metrics([self._row()], norm_weeks=4)
        assert metrics["norm_weeks"] == 4

    def test_revenue_per_week_computed(self, analyzer):
        rows = [
            self._row(label="A", revenue=10.0, week="2026-W16"),
            self._row(label="A", revenue=10.0, week="2026-W17"),
        ]
        art = analyzer.compute_metrics(rows)["top_articles_revenue"][0]
        assert art["revenue_per_week"] == 10.0  # 20 / 2
