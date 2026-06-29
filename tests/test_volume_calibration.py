"""Tests unitaires pour la calibration glissante des volumes par transaction."""

from datetime import date

import pytest

from stocks.volume_calibration import (
    MIN_VARIABLE_SALES,
    aggregate_window_consumption,
    calibrate_group,
    calibrate_volumes_in_items,
    item_calibratable,
)
from stocks.sumup_stocks import (
    build_stock_groups,
    build_sku_index,
    match_product_to_sku,
    prepare_enabled_stock_items,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wine_item(per_sale=0.15, unit="L", extra=None):
    """Article de vin (versé, calibrable par défaut) avec un état du stock."""
    item = {
        "stock_sku": "vin_rose",
        "label": "Vin rosé",
        "stock_label": "Vin rosé",
        "enabled": True,
        "unit": unit,
        "category": "bar",
        "is_stock_reference": True,
        "consumption_per_sale": per_sale,
        "sumup_match": {"name": "Vin", "variant": "Rosé"},
        "stock_state": {
            "stock_on_hand": 0.0,
            "stock_reserved": 0,
            "incoming_qty": 0,
            "incoming_eta": "",
            "last_inventory_date": "2026-01-01",
            "inventory_count_method": "manual",
            "stock_history": [],
        },
    }
    if extra:
        item.update(extra)
    return item


def _inv(d, counted):
    return {"type": "inventory", "date": d, "counted_qty": counted, "new_stock_on_hand": counted}


def _purchase(d, qty):
    return {"type": "purchase", "date": d, "qty_added": qty}


def _txn(d, name="Vin", variant="Rosé", qty=1):
    return {
        "timestamp": f"{d}T12:00:00Z",
        "status": "SUCCESSFUL",
        "products": [{"name": name, "description": variant, "quantity": qty}],
    }


def _prepare(raw_items):
    """Reproduit la préparation du pipeline (groupes + index)."""
    stock_items = prepare_enabled_stock_items(raw_items)
    groups = build_stock_groups(stock_items)
    index = build_sku_index(stock_items)
    return stock_items, groups, index


# ── item_calibratable ─────────────────────────────────────────────────────────

class TestItemCalibratable:
    def test_measure_unit_is_calibratable(self):
        assert item_calibratable({"unit": "L", "consumption_per_sale": 0.15})
        assert item_calibratable({"unit": "g", "consumption_per_sale": 10})
        assert item_calibratable({"stock_unit": "cl"})

    def test_count_unit_not_calibratable(self):
        assert not item_calibratable({"unit": "bouteille"})
        assert not item_calibratable({"unit": "canette"})
        assert not item_calibratable({"unit": "sachet"})
        assert not item_calibratable({"unit": "piece"})

    def test_explicit_flag_overrides(self):
        assert item_calibratable({"unit": "bouteille", "calibrate_volume": True})
        assert not item_calibratable({"unit": "L", "calibrate_volume": False})


# ── aggregate_window_consumption ──────────────────────────────────────────────

class TestAggregateWindow:
    def test_window_bounds_exclusive_start_inclusive_end(self):
        raw = [_wine_item()]
        _items, _groups, index = _prepare(raw)
        txns = [
            _txn("2026-05-01"),  # = start, exclu
            _txn("2026-05-02"),  # dans la fenêtre
            _txn("2026-05-10"),  # = end, inclus
            _txn("2026-05-11"),  # après, exclu
        ]
        res = aggregate_window_consumption(txns, index, date(2026, 5, 1), date(2026, 5, 10), match_product_to_sku)
        # 2 ventes comptées × 0.15
        assert res["vin_rose"]["total"] == pytest.approx(0.30)
        assert res["vin_rose"]["n_var_sales"] == 2

    def test_fixed_part_from_non_calibratable(self):
        raw = [
            _wine_item(),  # versé (variable)
            {  # pot exact, non calibrable
                "stock_sku": "vin_rose", "label": "Rose Pot", "enabled": True,
                "unit": "L", "category": "bar", "is_stock_reference": False,
                "consumption_per_sale": 1, "calibrate_volume": False,
                "sumup_match": {"name": "Vin", "variant": "Rose Pot"},
            },
        ]
        _items, _groups, index = _prepare(raw)
        txns = [
            _txn("2026-05-02", variant="Rosé", qty=2),       # variable : 2×0.15
            _txn("2026-05-03", variant="Rose Pot", qty=1),   # fixe : 1×1
        ]
        res = aggregate_window_consumption(txns, index, date(2026, 5, 1), date(2026, 5, 10), match_product_to_sku)
        assert res["vin_rose"]["total"] == pytest.approx(0.30 + 1.0)
        assert res["vin_rose"]["fixed"] == pytest.approx(1.0)
        assert res["vin_rose"]["n_var_sales"] == 2


# ── calibrate_group ───────────────────────────────────────────────────────────

class TestCalibrateGroup:
    def test_needs_two_inventories(self):
        raw = [_wine_item()]
        raw[0]["stock_state"]["stock_history"] = [_inv("2026-05-01", 10.0)]
        _items, groups, index = _prepare(raw)
        assert calibrate_group(groups[0], [], index, match_product_to_sku) is None

    def test_over_pour_increases_volume(self):
        # Compté 10 L le 01, 0 L le 10 ; aucun achat ⇒ conso réelle = 10 L.
        # 50 verres vendus × 0.15 = 7.5 L théoriques ⇒ on sert trop (ratio≈1.33).
        raw = [_wine_item(per_sale=0.15)]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 10.0),
            _inv("2026-05-10", 0.0),
        ]
        _items, groups, index = _prepare(raw)
        txns = [_txn(f"2026-05-0{(i % 8) + 2}", qty=1) for i in range(50)]

        summary = calibrate_group(groups[0], txns, index, match_product_to_sku, alpha=0.5)

        assert summary is not None
        assert summary["applied"] == 1
        raw_ref = groups[0]["reference_item"]["_raw_ref"]
        assert raw_ref["declared_consumption_per_sale"] == 0.15
        # Volume calibré strictement supérieur au déclaré (on sert trop).
        assert raw_ref["consumption_per_sale"] > 0.15
        # Pas au-delà de la borne +50 % sur un pas.
        assert raw_ref["consumption_per_sale"] <= 0.15 * 1.5
        calib = raw_ref["volume_calibration"]
        assert calib["last_calibrated_date"] == "2026-05-10"
        assert calib["current_factor"] > 1.0

    def test_under_pour_decreases_volume(self):
        # Conso réelle 5 L pour 50 verres théoriques à 0.15 (=7.5) ⇒ on sert trop peu.
        raw = [_wine_item(per_sale=0.15)]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 5.0),
            _inv("2026-05-10", 0.0),
        ]
        _items, groups, index = _prepare(raw)
        txns = [_txn(f"2026-05-0{(i % 8) + 2}", qty=1) for i in range(50)]

        calibrate_group(groups[0], txns, index, match_product_to_sku, alpha=0.5)
        raw_ref = groups[0]["reference_item"]["_raw_ref"]
        assert raw_ref["consumption_per_sale"] < 0.15

    def test_insufficient_sales_skipped(self):
        raw = [_wine_item(per_sale=0.15)]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 10.0),
            _inv("2026-05-10", 0.0),
        ]
        _items, groups, index = _prepare(raw)
        # Moins de MIN_VARIABLE_SALES ventes
        txns = [_txn("2026-05-02") for _ in range(MIN_VARIABLE_SALES - 1)]

        summary = calibrate_group(groups[0], txns, index, match_product_to_sku)
        raw_ref = groups[0]["reference_item"]["_raw_ref"]
        # Aucun ajustement appliqué, volume inchangé.
        assert "declared_consumption_per_sale" not in raw_ref
        assert raw_ref["consumption_per_sale"] == 0.15
        # Mais l'historique trace la raison.
        assert raw_ref["volume_calibration"]["history"][0]["applied"] is False

    def test_purchases_accounted_in_window(self):
        # 10 L le 01, achat +5 L le 05, 0 L le 10 ⇒ conso réelle = 15 L.
        raw = [_wine_item(per_sale=0.15)]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 10.0),
            _purchase("2026-05-05", 5.0),
            _inv("2026-05-10", 0.0),
        ]
        _items, groups, index = _prepare(raw)
        txns = [_txn(f"2026-05-0{(i % 8) + 2}", qty=1) for i in range(50)]

        calibrate_group(groups[0], txns, index, match_product_to_sku, alpha=0.5)
        entry = groups[0]["reference_item"]["_raw_ref"]["volume_calibration"]["history"][0]
        assert entry["actual_consumed"] == pytest.approx(15.0)

    def test_non_calibratable_reference_skipped(self):
        raw = [_wine_item(extra={"calibrate_volume": False})]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 10.0),
            _inv("2026-05-10", 0.0),
        ]
        _items, groups, index = _prepare(raw)
        txns = [_txn("2026-05-02") for _ in range(50)]
        assert calibrate_group(groups[0], txns, index, match_product_to_sku) is None

    def test_idempotent_second_run(self):
        raw = [_wine_item(per_sale=0.15)]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 10.0),
            _inv("2026-05-10", 0.0),
        ]
        _items, groups, index = _prepare(raw)
        txns = [_txn(f"2026-05-0{(i % 8) + 2}", qty=1) for i in range(50)]

        calibrate_group(groups[0], txns, index, match_product_to_sku, alpha=0.5)
        after_first = groups[0]["reference_item"]["_raw_ref"]["consumption_per_sale"]
        # Deuxième passe : le couple est déjà traité, pas de nouvel ajustement.
        summary2 = calibrate_group(groups[0], txns, index, match_product_to_sku, alpha=0.5)
        after_second = groups[0]["reference_item"]["_raw_ref"]["consumption_per_sale"]
        assert after_first == after_second
        assert summary2 is None or summary2.get("applied", 0) == 0


# ── calibrate_volumes_in_items (orchestrateur) ────────────────────────────────

class TestCalibrateVolumesInItems:
    def test_changed_flag_and_summaries(self):
        raw = [_wine_item(per_sale=0.15)]
        raw[0]["stock_state"]["stock_history"] = [
            _inv("2026-05-01", 10.0),
            _inv("2026-05-10", 0.0),
        ]
        stock_items, groups, index = _prepare(raw)
        txns = [_txn(f"2026-05-0{(i % 8) + 2}", qty=1) for i in range(50)]

        changed, summaries = calibrate_volumes_in_items(stock_items, groups, txns, index, match_product_to_sku)
        assert changed is True
        assert summaries and summaries[0]["stock_sku"] == "vin_rose"

    def test_no_inventory_no_change(self):
        raw = [_wine_item(per_sale=0.15)]
        stock_items, groups, index = _prepare(raw)
        changed, summaries = calibrate_volumes_in_items(stock_items, groups, [], index, match_product_to_sku)
        assert changed is False
        assert summaries == []
