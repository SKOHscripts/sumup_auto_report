#!/usr/bin/env python3
"""
Calibration glissante des volumes par transaction.
──────────────────────────────────────────────────
Les décomptes par vente (``consumption_per_sale``) sont saisis à la main :
un verre de vin « standard » à 15 cL, une pression déduite au litre, etc.
En pratique, les personnes au comptoir ne servent jamais exactement la
quantité déclarée, et le stock théorique dérive peu à peu du stock réel.

Ce module corrige cette dérive de façon **automatique et progressive** à
partir des « états des lieux » (comptages physiques) saisis dans l'Excel de
suivi (voir ``update_stock_from_purchases.py``). Entre deux états des lieux,
la consommation réelle d'un stock est connue exactement :

    conso_réelle = stock_compté_début + achats_période − stock_compté_fin

On la compare à la consommation théorique (ventes × ``consumption_per_sale``)
sur la même période, et on ajuste les volumes par transaction avec un
lissage glissant (moyenne mobile exponentielle), de sorte que :

  - un comptage isolé un peu faux ne fait pas sauter brutalement les volumes ;
  - sur quelques inventaires, les volumes convergent vers la réalité du bar.

Portée (« versés imprécis seulement ») : seuls les articles servis à une
quantité **mesurée** (unité L, cL, mL, g, kg…) dérivent. Les contenants
comptés à l'unité (bouteille, canette, sachet…) gardent un volume fixe et
servent de référence stable dans le calcul. Un article peut forcer son
comportement via le champ ``calibrate_volume`` (true/false) du catalogue.

La calibration ne modifie que ``consumption_per_sale`` dans
``stock_items.json`` ; la valeur déclarée d'origine est conservée dans
``declared_consumption_per_sale`` et l'historique des ajustements dans le
bloc ``volume_calibration`` de l'article de référence, pour information dans
le rapport.
"""

import logging
import re
from collections import defaultdict
from datetime import date, datetime

from utils.sumup_shared import normalize

log = logging.getLogger(__name__)

# ── Paramètres de calibration (réglages « équilibrés » par défaut) ─────────────

# Part du chemin parcouru vers la valeur mesurée à chaque état des lieux.
# 0.50 (« réactif ») ⇒ convergence rapide en ~2-3 inventaires : les volumes
# rattrapent vite la mesure physique, au prix d'une sensibilité un peu plus
# grande à un comptage isolé (atténuée par les bornes RATIO_*/STEP_*).
CALIBRATION_ALPHA = 0.50

# Bornes de bon sens sur le ratio brut conso_réelle / conso_théorique, pour
# neutraliser un comptage manifestement erroné avant le lissage.
RATIO_MIN = 0.5
RATIO_MAX = 2.0

# Borne dure sur le déplacement d'un volume en un seul pas (±50 %).
STEP_MIN = 0.5
STEP_MAX = 1.5

# Nombre minimal de ventes « calibrables » sur la période pour qu'un
# ajustement soit jugé significatif (évite de caler sur 1-2 ventes).
MIN_VARIABLE_SALES = 5

# Unités continues (versées / pesées) éligibles à la calibration par défaut.
MEASURE_UNITS = {"l", "cl", "ml", "dl", "g", "kg", "cl"}


# ── Détermination de l'éligibilité d'un article ───────────────────────────────

def _unit_of(item: dict) -> str:
    """Retourne l'unité de stock normalisée d'un article."""
    return normalize(item.get("stock_unit") or item.get("unit") or "")


def item_calibratable(item: dict) -> bool:
    """Indique si le volume d'un article doit être recalibré.

    Le champ explicite ``calibrate_volume`` (true/false) prime ; à défaut,
    un article est calibrable si son unité est une mesure continue (versé /
    pesé) plutôt qu'un comptage à l'unité.
    """
    explicit = item.get("calibrate_volume")
    if explicit is not None:
        return bool(explicit)
    return _unit_of(item) in MEASURE_UNITS


# ── Lecture des transactions (mêmes conventions que sumup_stocks) ──────────────

def _txn_date(txn: dict):
    """Retourne la date d'une transaction, ou None si illisible."""
    ts = txn.get("timestamp") or txn.get("transaction_date", "")
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _iter_sale_lines(txn: dict):
    """Itère les lignes de vente (name, variant, qty) d'une transaction.

    Ignore les transactions échouées/annulées. Replie sur ``product_summary``
    quand le détail produit est absent, comme le reste du pipeline.
    """
    status = (txn.get("status") or "").upper()
    if status in ("FAILED", "CANCELLED"):
        return

    products = txn.get("products") or []
    if not products:
        summary = txn.get("product_summary", "")
        for match in re.finditer(r"(\d+)\s*x\s*(.+?)(?:,|$)", summary):
            yield match.group(2).strip(), "", int(match.group(1))
        return

    for p in products:
        if not isinstance(p, dict):
            continue
        name = (p.get("name") or "").strip()
        variant = (p.get("description") or "").strip()
        try:
            qty = int(p.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1
        yield name, variant, qty


def _item_contributions(item: dict):
    """Itère les (stock_sku, per_sale, calibratable) consommés par un article.

    Couvre le stock principal et les éventuels ``also_consumes`` (ex. la crème
    de cassis d'un Kir). Pour un ``also_consumes``, l'éligibilité suit le
    flag de l'article serveur (le geste de service est le même).
    """
    primary_sku = item.get("stock_sku")
    if primary_sku:
        yield (
            primary_sku,
            float(item.get("consumption_per_sale", 1) or 1),
            item_calibratable(item),
        )
    for extra in item.get("also_consumes") or []:
        esku = extra.get("stock_sku")
        if esku:
            yield (
                esku,
                float(extra.get("consumption_per_sale", 0) or 0),
                item_calibratable(item),
            )


def aggregate_window_consumption(txns: list, sku_index: dict, start: date, end: date) -> dict:
    """Agrège la consommation théorique par stock_sku sur la fenêtre (start, end].

    Retourne ``{ stock_sku: {"total", "fixed", "n_var_sales"} }`` où ``fixed``
    est la part provenant d'articles non calibrables (volume considéré exact).
    """
    from stocks.sumup_stocks import match_product_to_sku  # pylint: disable=import-outside-toplevel

    res: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "fixed": 0.0, "n_var_sales": 0})

    for txn in txns:
        d = _txn_date(txn)
        if d is None or not start < d <= end:
            continue
        for name, variant, qty in _iter_sale_lines(txn):
            _sku, item = match_product_to_sku(name, variant, sku_index)
            if not item:
                continue
            for tgt_sku, per_sale, calibratable in _item_contributions(item):
                if per_sale <= 0:
                    continue
                amount = qty * per_sale
                bucket = res[tgt_sku]
                bucket["total"] += amount
                if calibratable:
                    bucket["n_var_sales"] += qty
                else:
                    bucket["fixed"] += amount

    return res


# ── Extraction des ancres physiques (états des lieux) ──────────────────────────

def _inventory_anchors(state: dict) -> list[tuple[date, float]]:
    """Retourne les états des lieux (date, quantité comptée) triés par date."""
    anchors = []
    for entry in state.get("stock_history") or []:
        if entry.get("type") != "inventory":
            continue
        try:
            d = date.fromisoformat(entry["date"])
        except (KeyError, ValueError):
            continue
        counted = entry.get("new_stock_on_hand")
        if counted is None:
            counted = entry.get("counted_qty")
        if counted is None:
            continue
        anchors.append((d, float(counted)))
    anchors.sort(key=lambda t: t[0])
    return anchors


def _purchases_in_window(state: dict, start: date, end: date) -> float:
    """Somme des quantités achetées (déjà en unités de stock) sur (start, end]."""
    total = 0.0
    for entry in state.get("stock_history") or []:
        if entry.get("type") != "purchase":
            continue
        try:
            d = date.fromisoformat(entry["date"])
        except (KeyError, ValueError):
            continue
        if start < d <= end:
            total += float(entry.get("qty_added", 0) or 0)
    return total


def _clamp(value: float, low: float, high: float) -> float:
    """Borne ``value`` dans [low, high]."""
    return max(low, min(high, value))


# ── Application de la calibration ──────────────────────────────────────────────

def _apply_step_to_sku(group: dict, stock_sku: str, step: float) -> None:
    """Multiplie les volumes calibrables des articles consommant ``stock_sku``.

    Mémorise au passage le volume déclaré d'origine dans
    ``declared_consumption_per_sale`` (article ou entrée ``also_consumes``).
    """
    for item in group["items"]:
        raw = item.get("_raw_ref", item)

        # Contribution principale
        if item.get("stock_sku") == stock_sku and item_calibratable(item):
            current = float(raw.get("consumption_per_sale", item.get("consumption_per_sale", 1)) or 1)
            if "declared_consumption_per_sale" not in raw:
                raw["declared_consumption_per_sale"] = current
            raw["consumption_per_sale"] = round(current * step, 6)
            item["consumption_per_sale"] = raw["consumption_per_sale"]

        # Contributions secondaires (also_consumes) vers ce stock_sku
        for raw_extra in raw.get("also_consumes") or []:
            if raw_extra.get("stock_sku") != stock_sku or not item_calibratable(item):
                continue
            current = float(raw_extra.get("consumption_per_sale", 0) or 0)
            if current <= 0:
                continue
            if "declared_consumption_per_sale" not in raw_extra:
                raw_extra["declared_consumption_per_sale"] = current
            raw_extra["consumption_per_sale"] = round(current * step, 6)


def _reference_factor(group: dict, stock_sku: str) -> float | None:
    """Retourne le facteur cumulé courant/déclaré du volume de référence."""
    ref = group["reference_item"]
    raw = ref.get("_raw_ref", ref)
    declared = raw.get("declared_consumption_per_sale")
    current = raw.get("consumption_per_sale")
    if declared in (None, 0) or current is None:
        return None
    try:
        return round(float(current) / float(declared), 4)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def calibrate_group(group: dict, txns: list, sku_index: dict,
                    alpha: float = CALIBRATION_ALPHA) -> dict | None:
    """Calibre les volumes d'un groupe de stock depuis ses états des lieux.

    Pour chaque couple d'états des lieux consécutifs non encore traité,
    compare consommation réelle et théorique puis ajuste les volumes par un
    pas lissé. Retourne un résumé de calibration, ou None si rien n'a changé.
    """
    ref = group["reference_item"]
    if not item_calibratable(ref):
        return None

    raw_ref = ref.get("_raw_ref", ref)
    state = raw_ref.get("stock_state") or {}
    anchors = _inventory_anchors(state)
    if len(anchors) < 2:
        return None  # Il faut deux comptages pour mesurer une consommation.

    stock_sku = group["stock_sku"]
    calib = dict(raw_ref.get("volume_calibration") or {})
    history = list(calib.get("history") or [])

    last_done = None
    if calib.get("last_calibrated_date"):
        try:
            last_done = date.fromisoformat(calib["last_calibrated_date"])
        except ValueError:
            last_done = None

    n_applied = 0

    for (d0, c0), (d1, c1) in zip(anchors, anchors[1:]):
        if d1 <= d0:
            continue
        if last_done is not None and d1 <= last_done:
            continue

        purchases = _purchases_in_window(state, d0, d1)
        actual = c0 + purchases - c1

        window = aggregate_window_consumption(txns, sku_index, d0, d1)
        bucket = window.get(stock_sku, {"total": 0.0, "fixed": 0.0, "n_var_sales": 0})
        theoretical_total = bucket["total"]
        theoretical_fixed = bucket["fixed"]
        theoretical_variable = theoretical_total - theoretical_fixed
        n_var_sales = bucket["n_var_sales"]

        entry = {
            "from_date": d0.isoformat(),
            "to_date": d1.isoformat(),
            "actual_consumed": round(actual, 3),
            "theoretical_total": round(theoretical_total, 3),
            "theoretical_fixed": round(theoretical_fixed, 3),
            "theoretical_variable": round(theoretical_variable, 3),
            "n_variable_sales": n_var_sales,
        }

        # Garde-fous : pas assez de signal pour caler de façon fiable.
        if theoretical_variable <= 0 or n_var_sales < MIN_VARIABLE_SALES:
            entry["applied"] = False
            entry["reason"] = "signal insuffisant"
            history.append(entry)
            last_done = d1
            continue

        raw_ratio = (actual - theoretical_fixed) / theoretical_variable
        clamped_ratio = _clamp(raw_ratio, RATIO_MIN, RATIO_MAX)
        step = _clamp(1.0 + alpha * (clamped_ratio - 1.0), STEP_MIN, STEP_MAX)

        _apply_step_to_sku(group, stock_sku, step)

        entry.update({
            "applied": True,
            "raw_ratio": round(raw_ratio, 4),
            "clamped_ratio": round(clamped_ratio, 4),
            "alpha": alpha,
            "step_multiplier": round(step, 4),
        })
        history.append(entry)
        last_done = d1
        n_applied += 1

        log.info(
            "Calibration %s [%s→%s] : reel=%.2f theo=%.2f ratio=%.2f pas=x%.3f",
            stock_sku, d0.isoformat(), d1.isoformat(),
            actual, theoretical_variable, raw_ratio, step,
        )

    if not history:
        return None

    calib["history"] = history
    calib["alpha"] = alpha
    if last_done is not None:
        calib["last_calibrated_date"] = last_done.isoformat()
    factor = _reference_factor(group, stock_sku)
    if factor is not None:
        calib["current_factor"] = factor
    raw_ref["volume_calibration"] = calib
    ref["volume_calibration"] = dict(calib)

    if n_applied == 0:
        # Historique enrichi (raisons), mais aucun volume modifié.
        return {"stock_sku": stock_sku, "applied": 0, "factor": factor}

    return {
        "stock_sku": stock_sku,
        "applied": n_applied,
        "factor": factor,
        "last_calibrated_date": calib.get("last_calibrated_date"),
    }


def calibrate_volumes_in_items(_stock_items: list, stock_groups: list,
                               txns: list, sku_index: dict,
                               alpha: float = CALIBRATION_ALPHA) -> tuple[bool, list]:
    """Calibre tous les groupes éligibles. Retourne (changed, résumés).

    ``changed`` est vrai dès qu'au moins un volume a été ajusté (donc qu'il
    faut re-sauvegarder le catalogue). Les résumés alimentent le rapport.
    """
    changed = False
    summaries = []

    for group in stock_groups:
        summary = calibrate_group(group, txns, sku_index, alpha=alpha)
        if summary is None:
            continue
        summaries.append(summary)
        if summary.get("applied", 0) > 0:
            changed = True

    if changed:
        log.info(
            "Calibration glissante : %d ajustement(s) de volume sur %d article(s).",
            sum(s.get("applied", 0) for s in summaries),
            sum(1 for s in summaries if s.get("applied", 0) > 0),
        )

    return changed, summaries
