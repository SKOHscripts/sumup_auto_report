#!/usr/bin/env python3
"""
SumUp - Rapport hebdomadaire de gestion des stocks
───────────────────────────────────────────────────
Usage :
  python sumup_stocks.py                          # 8 dernières semaines
  python sumup_stocks.py --no-mail                # PDF local, sans envoi email
  python sumup_stocks.py --weeks 12               # 12 semaines d'historique
  python sumup_stocks.py --mock mock_transactions.json
  python sumup_stocks.py --items stock_items.json --state stock_state.json

Automatisation via crontab (exemple : chaque lundi à 09:00) :
  0 9 * * 1 /usr/bin/python3 /chemin/vers/sumup_stocks.py >> /var/log/sumup_stocks.log 2>&1

Fichiers de configuration attendus dans le même répertoire :
  - stock_items.json  : catalogue des articles à suivre
  - stock_state.json  : état physique du stock à date

Fichiers générés :
  - rapport_stocks_YYYY-WNN.pdf
  - rapport_stocks_YYYY-WNN.csv
  - rapport_stocks_history_YYYY-WNN.csv
"""

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

# Permet l'exécution directe `python stocks/sumup_stocks.py` en plus de `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import requests
import fpdf as _fpdf
from fpdf import FPDF

from utils.mail_utils import (
    load_project_env,
    setup_memory_log_capture,
    send_email,
    )
from utils.sumup_shared import normalize, iso_week_label, week_start

# ─── Vérification version fpdf2 ───────────────────────────────────────────────


def _check_fpdf_version():
    """Vérifie que fpdf2 >= 2.5.2 est installé, sinon lève RuntimeError."""
    version = getattr(_fpdf, "__version__", "0")
    nums = []

    for part in version.split(".")[:3]:
        try:
            nums.append(int(part))
        except ValueError:
            break

    while len(nums) < 3:
        nums.append(0)

    if tuple(nums[:3]) < (2, 5, 2):
        raise RuntimeError(
            f"Version fpdf incompatible: {version}. "
            "Installez fpdf2>=2.5.2 (recommandé: fpdf2>=2.7,<3)."
            )


_check_fpdf_version()

DEFAULT_WEEKS = 4
PROJECTION_WEEKS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

load_project_env(
    required_vars=["SUMUP_API_KEY"],
    logger=log,
    )

_log_buffer, _log_handler = setup_memory_log_capture()
SUMUP_API_KEY = os.getenv("SUMUP_API_KEY")


# ─── 1. UTILITAIRES ───────────────────────────────────────────────────────────

def format_sumup_display(item: dict) -> str:
    """Retourne la chaîne d'affichage SumUp « Nom (Variante) » ou « Nom »."""
    sm = item.get("sumup_match", {})
    name = (sm.get("name") or item.get("label") or item.get("stock_sku") or "").strip()
    variant = (sm.get("variant") or "").strip()

    return f"{name} ({variant})" if variant else name


def fmt_num(value, decimals=2, width=6):
    """Formate un nombre en chaîne fixe ou retourne 'N/A' si None."""
    if value is None:
        return "N/A"

    return f"{float(value):{width}.{decimals}f}"


def load_stock_items_raw(path: Path) -> list:
    """Charge et retourne la liste brute JSON depuis stock_items.json."""
    if not path.exists():
        raise FileNotFoundError(f"stock_items.json introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("stock_items.json doit contenir une liste JSON")

    return items


def prepare_enabled_stock_items(raw_items: list) -> list:
    """Filtre les articles actifs et normalise leurs champs depuis la liste brute."""
    enabled = []

    for raw in raw_items:
        item = dict(raw)
        item["_raw_ref"] = raw
        item["stock_sku"] = item.get("stock_sku") or item["sku"]
        item["stock_label"] = item.get("stock_label") or item.get("label") or item["stock_sku"]
        item["stock_unit"] = item.get("stock_unit") or item.get("unit") or "piece"
        item["consumption_per_sale"] = float(
            item.get("consumption_per_sale", item.get("pack_size", 1) or 1)
        )
        item["is_stock_reference"] = bool(item.get("is_stock_reference")) or bool(item.get("stock_state"))

        if not item.get("enabled", True):
            continue

        enabled.append(item)

    log.info("Catalogue unifie : %s/%s article(s) actif(s) charge(s)", len(enabled), len(raw_items))

    return enabled


def save_stock_items(path: Path, raw_items: list):
    """Sauvegarde la liste brute dans stock_items.json via fichier temporaire."""
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def get_refresh_start_dt(stock_groups: list, fallback_start_dt: datetime) -> datetime:
    """Retourne la date de début la plus ancienne entre les ancres et le fallback."""
    anchor_dates = []

    for group in stock_groups:
        ref = group["reference_item"]
        state = ref.get("stock_state") or {}
        anchor = state.get("last_auto_update") or state.get("last_inventory_date")

        if not anchor:
            continue

        try:
            anchor_date = date.fromisoformat(anchor)
        except Exception:
            continue

        anchor_dates.append(datetime.combine(anchor_date, datetime.min.time(), tzinfo=timezone.utc))

    if not anchor_dates:
        return fallback_start_dt

    return min(fallback_start_dt, *anchor_dates)


def aggregate_stock_usage_since(txns: list, sku_index: dict, anchors_by_sku: dict, as_of: date) -> dict:
    """
    Agrège l'utilisation des stocks en une seule passe sur les transactions,
    en fonction d'une date d'ancrage spécifique à chaque stock_sku.
    Complexité : O(T) où T est le nombre de transactions.
    """
    usage_by_stock_sku = defaultdict(float)

    for txn in txns:
        status = (txn.get("status") or "").upper()

        if status in ("FAILED", "CANCELLED"):
            continue

        ts = txn.get("timestamp") or txn.get("transaction_date", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            txn_date = dt.date()
        except Exception:
            continue

        if txn_date > as_of:
            continue

        products = txn.get("products") or []

        # Cas de Fallback

        if not products:
            summary = txn.get("product_summary", "")
            sku, item = match_product_to_sku(summary, "", sku_index)

            if sku and item:
                stock_sku = item["stock_sku"]
                anchor = anchors_by_sku.get(stock_sku)
                # On vérifie l'ancre: l'ancre est exclue, le as_of est inclus

                if anchor and txn_date > anchor:
                    usage = float(item.get("consumption_per_sale", 1) or 1)
                    usage_by_stock_sku[stock_sku] += usage
                    log.warning("Fallback product_summary utilise pour %s (quantite 1 deduite).", stock_sku)

            continue

        # Cas Nominal

        for p in products:
            if not isinstance(p, dict):
                continue

            name = (p.get("name") or "").strip()
            variant = (p.get("description") or "").strip()

            try:
                qty = int(p.get("quantity") or 1)
            except Exception:
                qty = 1

            sku, item = match_product_to_sku(name, variant, sku_index)

            if not sku or not item:
                continue

            stock_sku = item["stock_sku"]
            anchor = anchors_by_sku.get(stock_sku)

            if anchor and txn_date > anchor:
                usage = qty * float(item.get("consumption_per_sale", 1) or 1)
                usage_by_stock_sku[stock_sku] += usage

    return usage_by_stock_sku


def refresh_stock_state_in_items(_stock_items, stock_groups, txns, sku_index, as_of=None):
    """
    Met à jour l'état du stock dans le catalogue unifié de manière optimisée.
    Complexité : O(G + T)
    """
    as_of = as_of or date.today()
    changed = False

    # --- Phase 1: Préparation des ancres ---
    anchors_by_sku = {}
    valid_groups = {}  # Pour garder le contexte lors de l'application

    for group in stock_groups:
        ref = group["reference_item"]
        state = ref.get("stock_state") or {}
        anchor_str = state.get("last_auto_update") or state.get("last_inventory_date")

        if not anchor_str:
            continue

        try:
            anchor_date = date.fromisoformat(anchor_str)
        except Exception:
            continue

        if anchor_date >= as_of:
            continue

        stock_sku = group["stock_sku"]
        anchors_by_sku[stock_sku] = anchor_date
        valid_groups[stock_sku] = (group, anchor_date)

    if not anchors_by_sku:
        return False

    # --- Phase 2: Agrégation unique (Single-Pass) ---
    usages = aggregate_stock_usage_since(
        txns=txns,
        sku_index=sku_index,
        anchors_by_sku=anchors_by_sku,
        as_of=as_of
    )

    # --- Phase 3: Application et Sauvegarde en mémoire ---

    for stock_sku, (group, anchor_date) in valid_groups.items():
        usage = usages.get(stock_sku, 0.0)

        if usage <= 0:
            continue

        ref = group["reference_item"]
        raw_ref = ref.get("_raw_ref", ref)
        state = dict(raw_ref.get("stock_state") or {})

        current_stock = float(state.get("stock_on_hand", 0) or 0)
        new_stock = max(0.0, current_stock - usage)

        history = list(state.get("stock_history") or [])
        history.append({
            "type": "auto_refresh",
            "from_date": anchor_date.isoformat(),
            "to_date": as_of.isoformat(),
            "consumed_qty": round(usage, 2),
            "previous_stock_on_hand": round(current_stock, 2),
            "new_stock_on_hand": round(new_stock, 2),
        })

        # Mutabilité sur le dict d'origine
        state["stock_on_hand"] = round(new_stock, 2)
        state["last_auto_update"] = as_of.isoformat()
        state["stock_history"] = history

        raw_ref["stock_state"] = state
        ref["stock_state"] = dict(state)
        changed = True

        log.info(
            "Maj auto stock %s: %.2f -> %.2f (conso %.2f depuis %s)",
            stock_sku, current_stock, new_stock, usage, anchor_date.isoformat(),
        )

    return changed

# ─── 2. CHARGEMENT DES FICHIERS DE CONFIGURATION ─────────────────────────────


def load_stock_items(path: Path) -> list:
    """Charge et normalise les articles actifs depuis stock_items.json."""
    if not path.exists():
        raise FileNotFoundError(f"stock_items.json introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    enabled = []

    for raw in items:
        if not raw.get("enabled", True):
            continue

        item = dict(raw)
        item["stock_sku"] = item.get("stock_sku") or item["sku"]
        item["stock_label"] = item.get("stock_label") or item.get("label") or item["stock_sku"]
        item["stock_unit"] = item.get("stock_unit") or item.get("unit") or "piece"
        item["consumption_per_sale"] = float(item.get("consumption_per_sale", item.get("pack_size", 1) or 1,
                                                      ))
        item["is_stock_reference"] = bool(item.get("is_stock_reference")) or bool(item.get("stock_state"))
        enabled.append(item)

    log.info("Catalogue unifie : %s/%s article(s) actif(s) charge(s)", len(enabled), len(items))

    return enabled


def load_stock_state(_path: Path) -> dict:
    """Avertit que le fichier stock_state.json séparé n'est plus utilisé."""
    log.warning("stock_state.json separe n'est plus utilise : l'etat est lu dans stock_items.json.")

    return {}


def build_stock_groups(stock_items: list) -> list:
    """Groupe les articles par stock_sku et désigne l'article de référence de chaque groupe."""
    grouped = defaultdict(list)

    for item in stock_items:
        grouped[item["stock_sku"]].append(item)

    groups = []

    for stock_sku, items in grouped.items():
        refs_with_state = [i for i in items if i.get("stock_state")]

        if len(refs_with_state) > 1:
            log.warning("%s: plusieurs lignes portent stock_state ; la premiere sera utilisee.", stock_sku)

        reference = refs_with_state[0] if refs_with_state else None

        if reference is None:
            reference = next((i for i in items if i.get("is_stock_reference")), None) or items[0]

        groups.append({
            "stock_sku": stock_sku,
            "reference_item": reference,
            "items": items,
            "state": dict(reference.get("stock_state") or {}),
            })

    return groups


def aggregate_weekly_stock_usage(stock_items: list, weekly_sales: dict, weeks_range: list) -> tuple:
    """Calcule la consommation et le nombre de ventes hebdomadaires par stock_sku."""
    weekly_usage = defaultdict(lambda: defaultdict(float))
    weekly_sales_count = defaultdict(lambda: defaultdict(int))

    seen_skus = set()

    for item in stock_items:
        stock_sku = item.get("stock_sku")

        if stock_sku in seen_skus:
            continue
        seen_skus.add(stock_sku)

        factor = float(item.get("consumption_per_sale", 1) or 1)

        for week_label in weeks_range:
            sold_qty = weekly_sales.get(item.get("stock_sku", ""), {}).get(week_label, 0)

            if sold_qty:
                weekly_usage[stock_sku][week_label] = sold_qty * factor
                weekly_sales_count[stock_sku][week_label] = sold_qty

    return weekly_usage, weekly_sales_count

# ─── 3. RÉCUPÉRATION ET ENRICHISSEMENT DES TRANSACTIONS ──────────────────────


def fetch_transactions(start: str, end: str, mock_file: str = None) -> list:
    """Récupère les transactions SumUp sur la période ou depuis un fichier mock."""
    if mock_file:
        log.info(" [MOCK] Lecture depuis '%s'", mock_file)
        with open(mock_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return data.get("items", data.get("transactions", []))

        return []

    headers = {"Authorization": f"Bearer {SUMUP_API_KEY}"}
    resp = requests.get(
        "https://api.sumup.com/v0.1/me/transactions/history",
        headers=headers,
        params={
            "limit": 5000,
            "order": "descending",
            "oldest_time": start,
            "newest_time": end,
            },
        timeout=20,
        )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", data.get("transactions", []))
    else:
        items = []
    log.info("Total brut recupere : %s transaction(s)", len(items))

    return items


def enrich_transactions(txns: list, headers: dict) -> list:
    """Enrichit chaque transaction avec le détail via GET /v0.1/me/transactions?id=."""
    log.info("Enrichissement de %s transaction(s)...", len(txns))
    enriched = []

    for t in txns:
        txn_id = t.get("id") or t.get("transaction_id")

        if not txn_id:
            enriched.append(t)

            continue
        try:
            resp = requests.get(
                "https://api.sumup.com/v0.1/me/transactions",
                headers=headers,
                params={"id": txn_id},
                timeout=10,
                )

            if resp.status_code == 200:
                detail = resp.json()

                if isinstance(detail, dict):
                    t = {**t, **{k: v for k, v in detail.items() if v is not None}}
            else:
                log.warning("reponse %s pour %s", resp.status_code, txn_id)
        except Exception as e:
            log.warning("Echec enrichissement %s : %s", txn_id, e)
        enriched.append(t)
        time.sleep(0.1)
    log.info("Enrichissement termine : %s transaction(s)", len(enriched))

    return enriched


# ─── 4. MAPPING TRANSACTIONS → SKU ───────────────────────────────────────────

def build_sku_index(stock_items: list) -> dict:
    """
    Construit un index de mapping (norm_name, norm_variant) -> item.
    La variant vide matche tous les produits du même nom sans variant déclarée.
    """
    index = {}

    for item in stock_items:
        sm = item.get("sumup_match", {})
        key = (
            normalize(sm.get("name", "")),
            normalize(sm.get("variant", "")),
            )
        index[key] = item

    return index


def match_product_to_sku(name: str, variant: str, sku_index: dict) -> tuple:
    """
    Retourne (sku, item) si trouvé, sinon (None, None).
    Priorité : (name, variant) exact > (name, '') pour les produits sans variante déclarée.
    """
    norm_name = normalize(name)
    norm_variant = normalize(variant)

    # Correspondance exacte name + variant
    key = (norm_name, norm_variant)

    if key in sku_index:
        return sku_index[key]["stock_sku"], sku_index[key]

    # Correspondance name seul si la config n'a pas de variant
    key_no_variant = (norm_name, "")

    if key_no_variant in sku_index:
        return sku_index[key_no_variant]["stock_sku"], sku_index[key_no_variant]

    # Correspondance partielle : le nom SumUp contient le nom config

    for (idx_name, idx_variant), item in sku_index.items():
        if idx_name and idx_name in norm_name:
            if not idx_variant or idx_variant in norm_variant:
                return item["stock_sku"], item

    return None, None


# ─── 5. AGRÉGATION HEBDOMADAIRE ──────────────────────────────────────────────

def aggregate_weekly_sales(txns: list, sku_index: dict, weeks_range: list) -> tuple:
    """
    Retourne :
      - weekly_sales  : dict { sku: { week_label: qty } }
      - unmapped_products : list de (name, variant, qty) non mappés
    """
    weekly_sales = defaultdict(lambda: defaultdict(int))
    unmapped = defaultdict(int)  # (name, variant) -> qty totale
    weeks_set = set(weeks_range)

    for txn in txns:
        status = (txn.get("status") or "").upper()

        if status in ("FAILED", "CANCELLED"):
            continue

        ts = txn.get("timestamp") or txn.get("transaction_date", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue

        week_label = iso_week_label(dt)

        if week_label not in weeks_set:
            continue

        products = txn.get("products") or []

        if not products:
            # Pas de produits détaillés : tenter depuis product_summary
            summary = txn.get("product_summary", "")

            for match in re.finditer(r'(\d+)\s*x\s*(.+?)(?:,|$)', summary):
                qty_s, name_s = int(match.group(1)), match.group(2).strip()
                sku, _item = match_product_to_sku(name_s, "", sku_index)

                if sku:
                    weekly_sales[sku][week_label] += qty_s

            continue

        for p in products:
            if not isinstance(p, dict):
                continue
            name = (p.get("name") or "").strip()
            variant = (p.get("description") or "").strip()
            try:
                qty = int(p.get("quantity") or 1)
            except Exception:
                qty = 1

            sku, _item = match_product_to_sku(name, variant, sku_index)

            if sku:
                weekly_sales[sku][week_label] += qty
            else:
                unmapped[(name, variant)] += qty

    unmapped_list = [
        {"name": k[0], "variant": k[1], "total_qty": v}

        for k, v in sorted(unmapped.items(), key=lambda x: -x[1])
        ]

    return weekly_sales, unmapped_list


# ─── 6. CALCUL DES INDICATEURS ───────────────────────────────────────────────

STATUS_ORDER = ["RISQUE RUPTURE", "A COMMANDER", "SURVEILLANCE", "OK", "N/A"]


def compute_dynamic_thresholds(item: dict, avg_rolling4: float, sales_7d: float) -> dict:
    """Calcule les seuils dynamiques (stock de sécurité, point de commande, cible)."""
    weekly_demand = max(float(avg_rolling4 or 0), 0.0)
    lead_time_days = int(item.get("supplier_lead_time_days", 7) or 7)
    lead_time_weeks = max(1, math.ceil(lead_time_days / 7))

    if weekly_demand <= 0:
        return {
            "weekly_demand": 0.0,
            "lead_time_weeks": lead_time_weeks,
            "safety_stock": 0.0,
            "reorder_point": 0.0,
            "target_stock": 0.0,
            }

    safety_stock = max(weekly_demand, sales_7d)
    reorder_point = (weekly_demand * lead_time_weeks) + safety_stock
    target_stock = weekly_demand * max(3, lead_time_weeks + 2)

    target_stock = max(target_stock, reorder_point)

    # Arrondi intelligent : entier si >= 1, sinon 2 décimales
    def smart_round(val):
        """Arrondit à l'entier si >= 1, sinon à 2 décimales."""
        return round(val) if val >= 1 else round(val, 2)

    return {
        "weekly_demand": round(weekly_demand, 2),
        "lead_time_weeks": lead_time_weeks,
        "safety_stock": smart_round(safety_stock),
        "reorder_point": smart_round(reorder_point),
        "target_stock": smart_round(target_stock),
        }


def compute_indicators(
    stock_group: dict, weekly_sales: dict, weekly_usage: dict,
    weekly_sales_count: dict, weeks_range: list,
) -> dict:
    """Calcule tous les indicateurs de stock (consommation, statut, projections) pour un groupe."""
    ref = stock_group["reference_item"]
    items = stock_group["items"]
    state = stock_group["state"]
    stock_sku = stock_group["stock_sku"]

    usage_by_week = weekly_usage.get(stock_sku, {})
    sales_by_week = weekly_sales_count.get(stock_sku, {})

    usage_series = [round(float(usage_by_week.get(w, 0)), 2) for w in weeks_range]
    sales_count_series = [int(sales_by_week.get(w, 0)) for w in weeks_range]

    total_used = sum(usage_series)
    n_weeks = len(weeks_range)
    n_zero_weeks = sum(1 for s in usage_series if s == 0)

    usage_7d = usage_series[-1] if usage_series else 0
    usage_28d = sum(usage_series[-4:]) if len(usage_series) >= 4 else sum(usage_series)

    avg_weekly = total_used / n_weeks if n_weeks > 0 else 0
    last4 = usage_series[-4:] if len(usage_series) >= 4 else usage_series
    avg_rolling4 = sum(last4) / len(last4) if last4 else 0

    prev_week_usage = usage_series[-2] if len(usage_series) >= 2 else None
    variation_pct = None

    if prev_week_usage is not None and prev_week_usage > 0:
        variation_pct = ((usage_7d - prev_week_usage) / prev_week_usage) * 100
    elif prev_week_usage == 0 and usage_7d > 0:
        variation_pct = 100.0

    proj_next_week = round(avg_rolling4, 1)
    proj_4_weeks = round(avg_rolling4 * PROJECTION_WEEKS, 1)

    stock_on_hand = float(state.get("stock_on_hand", 0) or 0)
    stock_reserved = float(state.get("stock_reserved", 0) or 0)
    incoming_qty = float(state.get("incoming_qty", 0) or 0)
    incoming_eta = state.get("incoming_eta") or None
    last_inventory_date = state.get("last_inventory_date") or "N/A"
    inventory_method = state.get("inventory_count_method") or "N/A"

    available_stock = stock_on_hand - stock_reserved

    thresholds = compute_dynamic_thresholds(ref, avg_rolling4, usage_7d)
    safety_stock = thresholds["safety_stock"]
    reorder_point = thresholds["reorder_point"]
    target_stock = thresholds["target_stock"]
    lead_time_weeks = thresholds["lead_time_weeks"]

    effective_stock_now = available_stock + incoming_qty

    coverage_weeks = None

    if avg_rolling4 > 0:
        coverage_weeks = effective_stock_now / avg_rolling4

    rupture_date = None

    if coverage_weeks is not None:
        rupture_dt = datetime.now() + timedelta(weeks=coverage_weeks)
        rupture_date = rupture_dt.date().isoformat()

    qty_to_order = max(0.0, float(target_stock) - effective_stock_now)

    if effective_stock_now <= 0:
        status = "RISQUE RUPTURE" if avg_rolling4 > 0 else "N/A"
    elif avg_rolling4 > 0 and coverage_weeks is not None and coverage_weeks < lead_time_weeks:
        status = "RISQUE RUPTURE"
    elif effective_stock_now <= reorder_point:
        status = "A COMMANDER"
    elif effective_stock_now <= max(safety_stock, reorder_point * 1.15):
        status = "SURVEILLANCE"
    else:
        status = "OK"

    linked_items = []

    for item in items:
        own_sales_series = [weekly_sales.get(item["stock_sku"], {}).get(w, 0) for w in weeks_range]
        sm = item.get("sumup_match", {})
        sm_name = (sm.get("name") or item.get("label") or item["stock_sku"]).strip()
        sm_variant = (sm.get("variant") or "").strip()

        linked_items.append({
            "stock_sku": item["stock_sku"],
            "label": item.get("label", item["stock_sku"]),
            "sumup_name": sm_name,
            "sumup_variant": sm_variant,
            "sumup_display": format_sumup_display(item),
            "unit": item.get("unit", "piece"),
            "consumption_per_sale": item.get("consumption_per_sale", 1),
            "sales_28d": sum(own_sales_series[-4:]) if len(own_sales_series) >= 4 else sum(own_sales_series),
            "sales_total": sum(own_sales_series),
            })

    return {
        "stock_sku": stock_sku,
        "label": ref.get("stock_label") or ref.get("label", stock_sku),
        "category": ref.get("category", ""),
        "unit": ref.get("stock_unit") or ref.get("unit", "piece"),
        "sumup_match": ref.get("sumup_match", {}),
        "linked_items": linked_items,
        "linked_items_count": len(linked_items),

        "stock_on_hand": stock_on_hand,
        "stock_reserved": stock_reserved,
        "available_stock": available_stock,
        "incoming_qty": incoming_qty,
        "incoming_eta": incoming_eta,
        "last_inventory_date": last_inventory_date,
        "inventory_method": inventory_method,

        "weekly_demand": thresholds["weekly_demand"],
        "lead_time_weeks": lead_time_weeks,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_stock": target_stock,

        "sales_series": usage_series,
        "usage_series": usage_series,
        "sales_count_series": sales_count_series,
        "weeks_range": weeks_range,
        "total_sold": round(total_used, 2),
        "total_used": round(total_used, 2),
        "sales_7d": round(usage_7d, 2),
        "usage_7d": round(usage_7d, 2),
        "sales_28d": round(usage_28d, 2),
        "usage_28d": round(usage_28d, 2),
        "avg_weekly": round(avg_weekly, 2),
        "avg_rolling4": round(avg_rolling4, 2),
        "variation_pct": round(variation_pct, 1) if variation_pct is not None else None,
        "n_zero_weeks": n_zero_weeks,

        "proj_next_week": proj_next_week,
        "proj_4_weeks": proj_4_weeks,
        "coverage_weeks": round(coverage_weeks, 1) if coverage_weeks is not None else None,
        "rupture_date": rupture_date,
        "qty_to_order": qty_to_order,
        "status": status,
        }


# ─── 7. GÉNÉRATION PDF ────────────────────────────────────────────────────────

PALETTE = {
    "accent": (0, 129, 138),
    "text_dark": (64, 59, 58),
    "text_mid": (107, 101, 100),
    "text_light": (158, 152, 151),
    "row_even": (237, 248, 249),
    "row_odd": (255, 255, 255),
    "divider": (210, 213, 220),
    "OK": (0, 129, 138),
    "SURVEILLANCE": (200, 134, 10),
    "A COMMANDER": (224, 90, 43),
    "RISQUE RUPTURE": (160, 38, 58),
    "N/A": (150, 150, 150),
    "status_bg": {
        "OK": (224, 247, 248),
        "SURVEILLANCE": (255, 243, 215),
        "A COMMANDER": (255, 228, 210),
        "RISQUE RUPTURE": (255, 215, 220),
        "N/A": (240, 240, 240),
        },
    }


class StockPDF(FPDF):
    """PDF du rapport de stocks : pages synthèse, articles et qualité données."""

    def __init__(self, week_label: str):
        """Initialise le PDF en portrait A4 avec la semaine de référence."""
        super().__init__(orientation="P", unit="mm", format="A4")
        self.week_label = week_label
        self.set_margins(14, 8, 14)
        self.set_auto_page_break(True, margin=16)

    def usable_width(self) -> float:
        """Retourne la largeur utilisable de la page (hors marges)."""
        return self.w - self.l_margin - self.r_margin

    def safe_str(self, text, max_len=999) -> str:
        """Nettoie le texte pour l'encodage latin-1 et tronque si nécessaire."""
        t = str(text or "-")
        replacements = {
            "€": "EUR",
            "—": "-",
            "–": "-",
            "’": "'",
            "“": '"',
            "”": '"',
            "\u00a0": " ",
            }

        for src, dst in replacements.items():
            t = t.replace(src, dst)
        t = t.encode("latin-1", errors="replace").decode("latin-1")

        return (t[:max_len - 3] + "...") if len(t) > max_len else t

    def status_color_for(self, status: str) -> tuple:
        """Retourne la couleur RGB associée au statut de stock."""
        return PALETTE.get(status, PALETTE["N/A"])

    def header(self):
        """Affiche la barre de titre et les informations d'en-tête."""
        self.set_font("Helvetica", "", 8)
        pw = self.usable_width()
        self.set_fill_color(*PALETTE["accent"])
        self.set_draw_color(*PALETTE["accent"])
        self.cell(0, 3, "", fill=True, border=0, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*PALETTE["text_dark"])
        self.cell(pw * 0.65, 9, " Rapport Gestion des Stocks",
                  border=0, fill=False, new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PALETTE["text_mid"])
        gen = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.cell(0, 9,
                  f"Semaine : {self.week_label}  |  Genere le {gen}",
                  border=0, fill=False, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        """Affiche le pied de page avec le numéro de page centré."""
        self.set_y(-13)
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_light"])
        self.cell(0, 10, f"SumUp - Rapport Stocks | Page {self.page_no()}", align="C")

    # ── Titre de section ────────────────────────────────────────────────────
    def section_title(self, title: str, color: tuple = None):
        """Insère un titre de section avec barre colorée et séparateur."""
        color = color or PALETTE["accent"]
        self.ln(2)
        self.set_fill_color(*color)
        y = self.get_y()
        self.rect(self.l_margin, y, 3, 7, style="F")
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(
            self.usable_width() - 5,
            7,
            self.safe_str(title.upper()),
            border=0,
            new_x="LMARGIN",
            new_y="NEXT",
            )
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.2)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)
        self.set_text_color(*PALETTE["text_dark"])

    # ── Bloc KPI 2 colonnes ─────────────────────────────────────────────────
    def kpi_block(self, kpis: list):
        """
        kpis : liste de (label, value) affichée sur 2 colonnes.
        """
        pw = self.usable_width()
        col_w = pw / 2

        for i, (label, value) in enumerate(kpis):
            if i % 2 == 0 and i > 0:
                self.ln(0)
            x_offset = self.l_margin + (col_w * (i % 2))
            self.set_xy(x_offset, self.get_y())
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_w * 0.55, 6, self.safe_str(label), border=0, align="L")
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*PALETTE["text_dark"])
            self.cell(col_w * 0.45, 6, self.safe_str(str(value)), border=0,
                      align="L", new_x="RIGHT" if i % 2 == 0 else "LMARGIN",
                      new_y="TOP" if i % 2 == 0 else "NEXT")

        if len(kpis) % 2 != 0:
            self.ln(6)
        self.ln(2)

    # ── Badge statut ────────────────────────────────────────────────────────
    def status_badge(self, status: str):
        """Affiche un badge coloré avec le statut de l'article."""
        color = self.status_color_for(status)
        bg = PALETTE["status_bg"].get(status, (240, 240, 240))
        self.set_fill_color(*bg)
        self.set_draw_color(*color)
        self.set_line_width(0.4)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        w = min(60, self.usable_width())
        self.cell(w, 8, f"  {status}  ", border=1, fill=True,
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_line_width(0.2)
        self.set_draw_color(*PALETTE["divider"])
        self.set_text_color(*PALETTE["text_dark"])

    # ── Tableau des ventes hebdomadaires ────────────────────────────────────

    def weekly_table(self, kpi: dict):
        """Affiche le tableau des ventes hebdomadaires avec moyennes glissantes et variations."""
        weeks = kpi["weeks_range"]
        sales = kpi.get("usage_series", [])
        sales_count = kpi.get("sales_count_series", [])

        pw = self.usable_width()
        n = len(weeks)

        if n == 0:
            return

        col_week = pw * 0.20
        col_count = pw * 0.15
        col_qty = pw * 0.20
        col_avg = pw * 0.22
        col_var = pw * 0.23
        row_h = 6.0
        head_h = 7.0

        # Calcul des moyennes glissantes et variations
        rows = []

        for i, (w, s, sc) in enumerate(zip(weeks, sales, sales_count)):
            last4 = sales[max(0, i - 3):i + 1]
            avg = sum(last4) / len(last4)

            if i > 0 and sales[i - 1] > 0:
                var = ((s - sales[i - 1]) / sales[i - 1]) * 100
                var_str = f"{var:+.0f}%"
            elif i > 0 and sales[i - 1] == 0 and s > 0:
                var_str = "+100%"
            elif i == 0:
                var_str = "-"
            else:
                var_str = "0%"
            rows.append((w, sc, s, avg, var_str))

        # En-tête
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        y = self.get_y()
        self.set_line_width(0.4)
        self.set_draw_color(*PALETTE["text_mid"])
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_line_width(0.2)
        self.set_draw_color(*PALETTE["divider"])
        self.cell(col_week, head_h, "Semaine", border=0, align="C")
        self.cell(col_count, head_h, "Nb Ventes", border=0, align="R")
        self.cell(col_qty, head_h, "Conso stock", border=0, align="R")
        self.cell(col_avg, head_h, "Moy. glissante", border=0, align="R")
        self.cell(col_var, head_h, "Variation", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

        # Lignes

        for i, (w, sc, s, avg, var_str) in enumerate(rows):
            if self.get_y() + row_h > self.h - self.b_margin:
                self.add_page()

            if i % 2 == 0:
                self.set_fill_color(*PALETTE["row_even"])
                self.rect(self.l_margin, self.get_y(), pw, row_h, style="F")
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_week, row_h, w, border="B", align="C")
            self.set_text_color(*PALETTE["text_dark"])
            self.cell(col_count, row_h, str(sc), border="B", align="R")
            self.cell(col_qty, row_h, str(s), border="B", align="R")
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_avg, row_h, f"{avg:.1f}", border="B", align="R")

            # Couleur variation
            try:
                var_val = float(var_str.replace("+", "").replace("%", ""))

                if var_val > 0:
                    self.set_text_color(0, 129, 138)
                elif var_val < 0:
                    self.set_text_color(160, 38, 58)
                else:
                    self.set_text_color(*PALETTE["text_mid"])
            except Exception:
                self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_var, row_h, var_str, border="B", align="R",
                      new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*PALETTE["text_dark"])
        self.ln(3)

    def weekly_graph(self, kpi: dict):
        """Génère et insère le graphique d'évolution du stock pour un article."""
        weeks = kpi["weeks_range"]
        sales = kpi["sales_series"]

        if not weeks or not sales:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(
                0, 6,
                self.safe_str("Aucune donnée hebdomadaire disponible."),
                new_x="LMARGIN", new_y="NEXT"
                )
            self.set_text_color(*PALETTE["text_dark"])

            return

        current_stock = float(kpi["available_stock"] or 0)
        incoming_qty = float(kpi["incoming_qty"] or 0)
        effective_stock_now = current_stock + incoming_qty
        avg_week = float(kpi["avg_rolling4"] or 0)

        # ── Reconstruction de la courbe historique hebdomadaire ──
        stock_curve = []
        running_stock = effective_stock_now

        for qty in reversed(sales):
            running_stock += float(qty or 0)

        for qty in sales:
            running_stock -= float(qty or 0)
            stock_curve.append(max(running_stock, 0.0))

        week_dates = []

        for lbl in weeks:
            try:
                year = int(lbl.split("-W")[0])
                week = int(lbl.split("-W")[1])
                week_dates.append(datetime.combine(week_start(year, week), datetime.min.time()))
            except Exception:
                continue

        if not week_dates:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(
                0, 6,
                self.safe_str("Impossible de construire l'axe temporel."),
                new_x="LMARGIN", new_y="NEXT"
                )
            self.set_text_color(*PALETTE["text_dark"])

            return

        def fmt_qty(v: float) -> str:
            """Formate une quantité : entier si valeur ronde, sinon 2 décimales."""
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))

            return f"{v:.2f}"

        now_dt = datetime.now().replace(microsecond=0)

        # ── Historique : on peut rejoindre 'aujourd'hui' seulement si la valeur
        # est identique au dernier point hebdo, pour éviter de raconter une fausse histoire.
        history_dates = list(week_dates)
        history_values = list(stock_curve)

        if history_values:
            last_hist_stock = float(history_values[-1])

            if now_dt > history_dates[-1] and abs(effective_stock_now - last_hist_stock) < 1e-9:
                history_dates.append(now_dt)
                history_values.append(last_hist_stock)

        # ── Tendance : départ au dernier point HEBDO, pas au jour courant ──
        start_idx = max(0, len(week_dates) - 4)
        trend_start_dt = week_dates[start_idx]
        trend_start_stock = float(stock_curve[start_idx]) if stock_curve else float(current_stock + incoming_qty)

        rupture_dt = None
        rupture_label = None

        future_dates = []
        future_stock = []

        if avg_week > 0 and trend_start_stock > 0:
            weeks_to_rupture = trend_start_stock / avg_week
            rupture_dt = trend_start_dt + timedelta(weeks=weeks_to_rupture)
            rupture_label = rupture_dt.strftime("%d/%m/%Y")

            n_future = max(4, int(math.floor(weeks_to_rupture)) + 2)

            for i in range(1, n_future + 1):
                dt_i = trend_start_dt + timedelta(weeks=i)
                val_i = trend_start_stock - (avg_week * i)
                future_dates.append(dt_i)
                future_stock.append(max(val_i, 0.0))

            # On ne garde pour la ligne que les points hebdo avant la rupture,
            # puis on ajoute le point précis de rupture à 0.
            trend_dates = [trend_start_dt]
            trend_values = [trend_start_stock]

            for dt_i, val_i in zip(future_dates, future_stock):
                if dt_i < rupture_dt:
                    trend_dates.append(dt_i)
                    trend_values.append(val_i)

            trend_dates.append(rupture_dt)
            trend_values.append(0.0)
        else:
            # Pas de consommation exploitable : on trace une ligne plate sur 4 semaines
            future_dates = [trend_start_dt + timedelta(weeks=i) for i in range(1, 5)]
            future_stock = [trend_start_stock for _ in future_dates]
            trend_dates = [trend_start_dt] + future_dates
            trend_values = [trend_start_stock] + future_stock

        safety_stock = float(kpi.get("safety_stock") or 0)
        reorder_point = float(kpi.get("reorder_point") or 0)
        target_stock = float(kpi.get("target_stock") or 0)

        fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=160)

        # Historique
        ax.plot(
            history_dates,
            history_values,
            color="#00818A",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Historique de stock",
            )

        # Tendance
        ax.plot(
            trend_dates,
            trend_values,
            color="#E05A2B",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=3.5,
            label="Tendance",
            )

        # Seuils
        ax.axvspan(week_dates[start_idx], week_dates[-1], color="#B3E0E3", alpha=0.18, label="Période de tendance")
        ax.axhspan(0, safety_stock, color="#FFE5C8", alpha=0.35)
        ax.axhline(safety_stock, color="#E05A2B", linestyle=":", linewidth=1.3, label="Stock de sécurité")
        ax.axhline(reorder_point, color="#FFA70B", linestyle=":", linewidth=1.3, label="Point de commande")

        plotted_values = (
            [1.0]
            + [float(v) for v in history_values]
            + [float(v) for v in trend_values]
            + [target_stock]
            + [reorder_point]
            + [safety_stock]
            )
        ymax = max(plotted_values) if plotted_values else 1.0
        ymax = max(ymax, 1.0)

        # Barre verticale exacte au point où la tendance touche 0

        if rupture_dt is not None:
            # ax.axvline(rupture_dt, color="#990000", linestyle="--", linewidth=1.2)
            label_y = ymax * 0.18 if ymax > 0 else 1.0
            ax.annotate(
                f"Rupture estimee\n{rupture_label}",
                xy=(rupture_dt, 0.0),
                xytext=(rupture_dt + timedelta(days=2), label_y),
                fontsize=8,
                color="#403B3A",
                arrowprops={"arrowstyle": "->", "color": "#403B3A", "lw": 1},
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#403B3A", "alpha": 0.9},
                )

        # Etiquettes historique

        for x, y in zip(week_dates, stock_curve):
            ax.annotate(
                fmt_qty(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color="#00818A",
                )

        # Etiquettes projection : seulement sur les ticks hebdo de projection

        for x, y in zip(future_dates, future_stock):
            ax.annotate(
                fmt_qty(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color="#C8860A",
                )

            ax.set_title("Evolution du stock et tendance", fontsize=11)
        ax.set_ylabel(f"Quantite [{kpi.get('unit') or 'S.U.'}]")

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-W%W"))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)

        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

        ax.set_ylim(0, ymax * 1.20)

        xmin = week_dates[0] - timedelta(days=2)
        xmax_candidates = [
            future_dates[-1] + timedelta(days=4) if future_dates else trend_start_dt + timedelta(weeks=4)
        ]

        if rupture_dt is not None:
            xmax_candidates.append(rupture_dt + timedelta(days=4))
        xmax = max(xmax_candidates)
        ax.set_xlim(xmin, xmax)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)

        chart_w = self.usable_width()
        chart_h = 72
        y0 = self.get_y()

        if y0 + chart_h > self.h - self.b_margin:
            self.add_page()
            y0 = self.get_y()

        self.image(buf, x=self.l_margin, y=y0, w=chart_w, h=chart_h)
        self.set_y(y0 + chart_h + 4)

# ─── Page 1 : Synthèse globale ────────────────────────────────────────────────


def render_page_summary(pdf: StockPDF, all_kpis: list, week_label: str, _weeks_range: list):
    """Génère la page de synthèse globale avec KPIs et tableau de statuts."""
    pdf.add_page()
    pw = pdf.usable_width()

    # KPIs globaux
    n_total = len(all_kpis)
    n_alert = sum(1 for k in all_kpis if k["status"] in ("SURVEILLANCE", "A COMMANDER", "RISQUE RUPTURE"))
    n_order = sum(1 for k in all_kpis if k["status"] == "A COMMANDER")
    n_rupture = sum(1 for k in all_kpis if k["status"] == "RISQUE RUPTURE")

    pdf.section_title(f"Synthese globale — Semaine {week_label}")
    pdf.kpi_block([
        ("Articles suivis", n_total),
        ("Articles en alerte", n_alert),
        ("Articles a commander", n_order),
        ("Risques de rupture", n_rupture),
        ])

    # Tableau statuts
    pdf.section_title("Etat des articles", PALETTE["accent"])
    # col_sku = pw * 0.22
    col_lbl = pw * 0.32
    col_unit = pw * 0.1
    col_stk = pw * 0.1
    col_cov = pw * 0.1
    col_cmd = pw * 0.12
    col_sta = pw * 0.22
    head_h = 7.0
    row_h = 6.5

    # En-tête tableau
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*PALETTE["text_mid"])
    y = pdf.get_y()
    pdf.set_line_width(0.4)
    pdf.set_draw_color(*PALETTE["text_mid"])
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(*PALETTE["divider"])
    # pdf.cell(col_sku, head_h, "SKU", border=0, align="L")
    pdf.cell(col_lbl, head_h, "Libelle", border=0, align="L")
    pdf.cell(col_unit, head_h, "Unité", border=0, align="R")
    pdf.cell(col_stk, head_h, "Stock dispo", border=0, align="R")
    pdf.cell(col_cov, head_h, "Couverture", border=0, align="R")
    pdf.cell(col_cmd, head_h, "A commander", border=0, align="R")
    pdf.cell(col_sta, head_h, "Statut", border=0, align="C",
             new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)

    # Tri par sévérité
    sorted_kpis = sorted(all_kpis, key=lambda k: STATUS_ORDER.index(k["status"]))

    for i, kpi in enumerate(sorted_kpis):
        if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()

        if i % 2 == 0:
            pdf.set_fill_color(*PALETTE["row_even"])
            pdf.rect(pdf.l_margin, pdf.get_y(), pw, row_h, style="F")

        status = kpi["status"]
        status_color = pdf.status_color_for(status)
        cov = f"{kpi['coverage_weeks']:.1f} sem." if kpi['coverage_weeks'] is not None else "N/A"

        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*PALETTE["text_mid"])
        # pdf.cell(col_sku, row_h, pdf.safe_str(kpi["stock_sku"], 20), border="B", align="L")
        pdf.set_text_color(*PALETTE["text_dark"])
        pdf.cell(col_lbl, row_h, pdf.safe_str(kpi["label"], 70), border="B", align="L")
        pdf.cell(col_unit, row_h, f"{str(kpi['unit'])}", border="B", align="R")
        pdf.cell(col_stk, row_h, f"{str(kpi['available_stock'])}", border="B", align="R")
        pdf.cell(col_cov, row_h, cov, border="B", align="R")
        pdf.cell(col_cmd, row_h, str(kpi["qty_to_order"]), border="B", align="R")
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*status_color)
        pdf.cell(col_sta, row_h, status, border="B", align="C",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(*PALETTE["text_dark"])
    pdf.ln(4)


# ─── Pages article ────────────────────────────────────────────────────────────


def render_article_page(pdf: StockPDF, kpi: dict):
    """Génère la page détaillée d'un article (KPIs, graphique, tableau hebdomadaire)."""
    pdf.add_page()

    pdf.section_title(f"Stock : {kpi['label']}")

    linked_text = ", ".join(
        it.get("sumup_display") or it.get("label") or it.get("stock_sku")

        for it in kpi.get("linked_items", [])
        )
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*PALETTE["text_mid"])
    pdf.multi_cell(0, 5, pdf.safe_str(f"Articles SumUp relies ({kpi['linked_items_count']}) : {linked_text}", 220))
    pdf.set_text_color(*PALETTE["text_dark"])
    pdf.ln(1)

    pdf.kpi_block([
        # ("Categorie", kpi["category"]),
        # ("Variante SumUp", variant_str),
        # ("Stock SKU", kpi["stock_sku"]),
        # ("Articles relies", kpi["linked_items_count"]),
        ("Unite stock", kpi["unit"]),
        (f"Stock disponible [{kpi['unit']}]", kpi["available_stock"]),
        (f"Stock arrivant [{kpi['unit']}]", kpi["incoming_qty"]),
        ("ETA reappro", kpi["incoming_eta"] or "N/A"),
        (f"Stock securite (calculé) [{kpi['unit']}]", kpi["safety_stock"]),
        (f"Point de commande (calculé) [{kpi['unit']}]", kpi["reorder_point"]),
        (f"Stock cible (calculé) [{kpi['unit']}]", kpi["target_stock"]),
        ("Dernier inventaire", kpi["last_inventory_date"]),
        ])

    # Badge statut
    pdf.status_badge(kpi["status"])

    # ── Bloc KPIs ──
    pdf.section_title("Indicateurs cles")
    # var = f"{kpi['variation_pct']:+.1f}%" if kpi["variation_pct"] is not None else "N/A"
    cov = f"{kpi['coverage_weeks']:.1f} sem." if kpi["coverage_weeks"] is not None else "N/A"
    pdf.kpi_block([
        # ("Ventes 7 jours", kpi["sales_7d"]),
        (f"Conso 28 jours [{kpi['unit']}]", kpi["usage_28d"]),
        (f"Moyenne hebdo conso [{kpi['unit']}]", kpi["avg_weekly"]),
        (f"Moy. glissante 4 sem. [{kpi['unit']}]", kpi["avg_rolling4"]),
        # ("Projection sem. suiv.", kpi["proj_next_week"]),
        # ("Projection vente 4 sem.", kpi["proj_4_weeks"]),
        ("Couverture estimee", cov),
        ("Date rupture estimee", kpi["rupture_date"] or "N/A"),
        (f"Qte a commander [{kpi['unit']}]", kpi["qty_to_order"]),
        # ("Variation S vs S-1", var),
        # ("Sem. sans vente", kpi["n_zero_weeks"]),
        (f"Total consomme (periode) [{kpi['unit']}]", kpi["total_used"]),
        ])

    # ── Tableau hebdomadaire ──
    pdf.section_title("Evolution du stock")
    pdf.weekly_graph(kpi)
    pdf.weekly_table(kpi)


# ─── Dernière page : qualité des données ─────────────────────────────────────

def render_data_quality_page(pdf: StockPDF, unmapped: list, all_kpis: list):
    """Génère la page de qualité des données (produits non mappés, dates d'inventaire)."""
    pdf.add_page()
    pdf.section_title("Qualite des donnees", PALETTE["text_mid"])

    pw = pdf.usable_width()

    # Articles non mappés
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*PALETTE["text_dark"])
    pdf.cell(0, 6, f"Produits SumUp non mappes : {len(unmapped)}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    if unmapped:
        col_name = pw * 0.45
        col_var = pw * 0.35
        col_qty = pw * 0.20
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*PALETTE["text_mid"])
        pdf.cell(col_name, 6, "Nom produit", border=0, align="L")
        pdf.cell(col_var, 6, "Variante", border=0, align="L")
        pdf.cell(col_qty, 6, "Qte totale", border=0, align="R",
                 new_x="LMARGIN", new_y="NEXT")

        for i, u in enumerate(unmapped):
            if pdf.get_y() + 5.5 > pdf.h - pdf.b_margin:
                break

            if i % 2 == 0:
                pdf.set_fill_color(*PALETTE["row_even"])
                pdf.rect(pdf.l_margin, pdf.get_y(), pw, 5.5, style="F")
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*PALETTE["text_dark"])
            pdf.cell(col_name, 5.5, pdf.safe_str(u["name"], 40), border="B", align="L")
            pdf.set_text_color(*PALETTE["text_mid"])
            pdf.cell(col_var, 5.5, pdf.safe_str(u["variant"], 30), border="B", align="L")
            pdf.cell(col_qty, 5.5, str(u["total_qty"]), border="B", align="R",
                     new_x="LMARGIN", new_y="NEXT")

        # if len(unmapped) > 30:
        #     pdf.set_font("Helvetica", "I", 7)
        #     pdf.set_text_color(*PALETTE["text_mid"])
        #     pdf.cell(0, 5, f"... et {len(unmapped) - 30} autre(s) non affiché(s)",
        #              new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*PALETTE["text_mid"])
        pdf.cell(0, 6, "Aucun produit non mappe.", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    # Date dernier inventaire par article
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*PALETTE["text_dark"])
    pdf.cell(0, 6, "Dernier inventaire connu par article :",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    for kpi in all_kpis:
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*PALETTE["text_mid"])
        pdf.cell(pw * 0.55, 5.5, pdf.safe_str(kpi["label"], 40), border=0, align="L")
        pdf.set_text_color(*PALETTE["text_dark"])
        pdf.cell(pw * 0.30, 5.5, str(kpi["last_inventory_date"]), border=0, align="L")
        pdf.cell(pw * 0.15, 5.5, kpi["inventory_method"], border=0, align="L",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(*PALETTE["text_dark"])


# ─── Génération complète du PDF ───────────────────────────────────────────────

def generate_pdf(all_kpis: list, unmapped: list, week_label: str, weeks_range: list,
                 path: str) -> str:
    """Génère le PDF complet (synthèse + pages articles + qualité) et le sauvegarde."""
    pdf = StockPDF(week_label)
    render_page_summary(pdf, all_kpis, week_label, weeks_range)

    for kpi in all_kpis:
        render_article_page(pdf, kpi)
    render_data_quality_page(pdf, unmapped, all_kpis)
    pdf.output(path)
    log.info("PDF genere -> %s", path)

    return path


# ─── 8. EXPORTS CSV ───────────────────────────────────────────────────────────

def export_csv_summary(all_kpis: list, path: str):
    """Exporte le CSV de synthèse des indicateurs par article."""
    fields = [
        "stock_sku", "label", "category", "unit",
        "available_stock", "incoming_qty", "incoming_eta",
        "safety_stock", "reorder_point", "target_stock",
        "usage_7d", "usage_28d", "avg_weekly", "avg_rolling4",
        "proj_next_week", "proj_4_weeks",
        "coverage_weeks", "rupture_date", "qty_to_order",
        "variation_pct", "n_zero_weeks", "total_used",
        "status", "last_inventory_date", "inventory_method",
        "linked_items_count",
        ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_kpis)
    log.info("CSV synthese -> %s", path)


def export_csv_history(all_kpis: list, path: str):
    """Exporte le CSV historique des consommations hebdomadaires par article."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stock_sku", "label", "week", "qty_used", "unit"])

        for kpi in all_kpis:
            for w, s in zip(kpi["weeks_range"], kpi.get("usage_series", kpi["sales_series"])):
                writer.writerow([kpi["stock_sku"], kpi["label"], w, s, kpi["unit"]])

    log.info("CSV historique -> %s", path)


# ─── 9. ENVOI EMAIL ───────────────────────────────────────────────────────────

def send_stock_email(
    weeks: str,
    pdf_path: str,
    _csv_path: str,
    _hist_path: str,
    week_label: str,
    all_kpis: list,
        ):
    """Compose et envoie l'email de rapport stocks avec le PDF en pièce jointe."""
    n_alert = sum(1 for k in all_kpis if k["status"] in ("SURVEILLANCE", "A COMMANDER", "RISQUE RUPTURE"))
    to_order = [k for k in all_kpis if k["status"] in ("A COMMANDER", "RISQUE RUPTURE")]

    order_lines = ""

    if to_order:
        order_lines = "\nArticles a commander :\n"

        for k in to_order:
            order_lines += (
                f"  - {k['label']} [{k['stock_sku']}] : {k['qty_to_order']} {k['unit']}(s)  "
                f"- Statut : {k['status']}\n"
                )
    else:
        order_lines = "\nAucun article a commander cette semaine.\n"

    body = f"""\
Bonjour,

Veuillez trouver en piece jointe le rapport hebdomadaire de gestion des stocks
pour la semaine {week_label}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUME - Semaine {week_label}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Durée d’historique  : {weeks} semaines
Articles suivis     : {len(all_kpis)}
Articles en alerte  : {n_alert}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cordialement,
Corentin via sumup_stocks.py
    """
    subject = (
        f"Rapport Stocks SumUp - {week_label} "
        f"({len(all_kpis)} articles, {n_alert} alerte(s))"
        )
    attachments = [pdf_path]
    # if csv_path and Path(csv_path).exists():
    #     attachments.append(csv_path)
    # if hist_path and Path(hist_path).exists():
    #     attachments.append(hist_path)

    send_email(
        subject=subject,
        body=body,
        attachments=attachments,
        mailing_list="all_ca",
        logger=log,
        )


# ─── 10. PIPELINE PRINCIPAL ───────────────────────────────────────────────────

def run_stock_report(
    weeks: int = DEFAULT_WEEKS,
    send_mail: bool = True,
    mock_file: str = None,
    items_file: Path = None,
    state_file: Path = None,
):
    """Exécute le pipeline complet : chargement, API, agrégation, PDF, CSV, email."""
    items_file = items_file or BASE_DIR / "stocks" / "stock_items.json"

    now = datetime.now(timezone.utc)
    end_dt = now
    analysis_start_dt = end_dt - timedelta(weeks=weeks)

    weeks_range = []
    cursor = analysis_start_dt
    seen = set()
    while cursor <= end_dt:
        lbl = iso_week_label(cursor)
        if lbl not in seen:
            weeks_range.append(lbl)
            seen.add(lbl)
        cursor += timedelta(days=7)
    weeks_range = sorted(set(weeks_range))

    current_week = iso_week_label(now)
    log.info("== Rapport Stocks SumUp == %s semaines | Semaine courante : %s", weeks, current_week)

    log.info("Étape 1/6 - Chargement du catalogue unifie…")
    raw_items = load_stock_items_raw(items_file)
    stock_items = prepare_enabled_stock_items(raw_items)
    stock_groups = build_stock_groups(stock_items)
    sku_index = build_sku_index(stock_items)

    if state_file:
        log.info("Le fichier stock_state.json separe est ignore : stock_state est lu depuis le catalogue unifie.")

    fetch_start_dt = get_refresh_start_dt(stock_groups, analysis_start_dt)
    fetch_start = fetch_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info(
        "Etape 2/6 - Recuperation des transactions... "
        "(fenetre analyse=%s..%s | fenetre fetch=%s..%s)",
        analysis_start_dt.date(), end_dt.date(),
        fetch_start_dt.date(), end_dt.date(),
    )
    headers_api = {"Authorization": f"Bearer {SUMUP_API_KEY}"}
    all_txns = fetch_transactions(fetch_start, fetch_end, mock_file=mock_file)
    if not mock_file:
        all_txns = enrich_transactions(all_txns, headers_api)

    log.info("Étape 3/6 - Agregation hebdomadaire…")
    weekly_sales, unmapped = aggregate_weekly_sales(all_txns, sku_index, weeks_range)
    weekly_usage, weekly_sales_count = aggregate_weekly_stock_usage(stock_items, weekly_sales, weeks_range)
    if unmapped:
        log.warning("%s produit(s) SumUp non mappe(s) au catalogue", len(unmapped))

    log.info("Étape 4/6 - Mise a jour automatique des stocks…")
    stocks_updated = refresh_stock_state_in_items(
        stock_items,
        stock_groups=stock_groups,
        txns=all_txns,
        sku_index=sku_index,
        as_of=now.date(),
    )

    if stocks_updated:
        save_stock_items(items_file, raw_items)
        log.info("Catalogue mis a jour : %s", items_file)

        stock_items = prepare_enabled_stock_items(raw_items)
        stock_groups = build_stock_groups(stock_items)
        sku_index = build_sku_index(stock_items)
    else:
        log.info("Aucune mise a jour automatique du stock necessaire.")

    log.info("Étape 5/6 - Calcul des indicateurs centralises…")
    all_kpis = []
    for group in stock_groups:
        kpi = compute_indicators(group, weekly_sales, weekly_usage, weekly_sales_count, weeks_range)
        all_kpis.append(kpi)
        log.info(
            " %-30s | stock=%s | vendu=%s | statut=%s",
            kpi['stock_sku'], fmt_num(kpi['available_stock']),
            fmt_num(kpi['total_sold']), kpi['status'],
        )

    log.info("Étape 6/6 - Generation des fichiers…")
    safe_week = current_week.replace("-", "_")
    pdf_path = str(BASE_DIR / f"rapport_stocks_{safe_week}.pdf")
    csv_path = str(BASE_DIR / f"rapport_stocks_{safe_week}.csv")
    hist_path = str(BASE_DIR / f"rapport_stocks_history_{safe_week}.csv")

    generate_pdf(all_kpis, unmapped, current_week, weeks_range, pdf_path)
    export_csv_summary(all_kpis, csv_path)
    export_csv_history(all_kpis, hist_path)

    if send_mail:
        send_stock_email(weeks, pdf_path, csv_path, hist_path, current_week, all_kpis)
    else:
        log.info("Envoi email ignore (--no-mail).")

    log.info("══ Termine ══")
    return all_kpis, unmapped


# ─── 11. CLI ──────────────────────────────────────────────────────────────────

def main():
    """Point d'entrée CLI : parse les arguments et lance run_stock_report."""
    parser = argparse.ArgumentParser(
        description="Rapport hebdomadaire de gestion des stocks SumUp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        )
    parser.add_argument(
        "--weeks", type=int, default=DEFAULT_WEEKS,
        help=f"Nombre de semaines d'historique (défaut : {DEFAULT_WEEKS})",
        )
    parser.add_argument(
        "--no-mail", action="store_true",
        help="Génère les fichiers sans envoyer l'email",
        )
    parser.add_argument(
        "--mock", metavar="FICHIER",
        help="Utilise un fichier JSON local à la place de l'API SumUp",
        )
    parser.add_argument(
        "--items", metavar="FICHIER", default=None,
        help="Chemin vers stock_items.json (défaut : ./stock_items.json)",
        )
    parser.add_argument(
        "--state", metavar="FICHIER", default=None,
        help="Chemin vers stock_state.json (défaut : ./stock_state.json)",
        )
    args = parser.parse_args()

    run_stock_report(
        weeks=args.weeks,
        send_mail=not args.no_mail,
        mock_file=args.mock or os.getenv("SUMUP_MOCK_FILE") or None,
        items_file=Path(args.items) if args.items else None,
        state_file=Path(args.state) if args.state else None,
        )


if __name__ == "__main__":
    main()
