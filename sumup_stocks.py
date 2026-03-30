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
import json
import logging
import os
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
import matplotlib.pyplot as plt
import math

import requests
import fpdf as _fpdf
from fpdf import FPDF

from mail_utils import (
    load_project_env,
    setup_memory_log_capture,
    send_email,
    build_log_footer,
)


# ─── Vérification version fpdf2 ───────────────────────────────────────────────
def _check_fpdf_version():
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

DEFAULT_WEEKS = 8
PROJECTION_WEEKS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_project_env(
    env_file=ENV_FILE,
    required_vars=["SUMUP_API_KEY"],
    logger=log,
)

_log_buffer, _log_handler = setup_memory_log_capture()
SUMUP_API_KEY = os.getenv("SUMUP_API_KEY")


# ─── 1. UTILITAIRES ───────────────────────────────────────────────────────────

def remove_accents(text: str) -> str:
    if not text:
        return ""

    return "".join(
        c for c in unicodedata.normalize("NFD", text)

        if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return remove_accents((text or "").lower()).strip()


def iso_week_label(dt: datetime) -> str:
    """Retourne le label ISO de la semaine, ex: '2026-W13'."""
    y, w, _ = dt.isocalendar()

    return f"{y}-W{w:02d}"


def week_start(year: int, week: int) -> date:
    """Retourne le lundi de la semaine ISO donnée."""
    jan4 = date(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.isoweekday() - 1)

    return start_of_week1 + timedelta(weeks=week - 1)


# ─── 2. CHARGEMENT DES FICHIERS DE CONFIGURATION ─────────────────────────────

def load_stock_items(path: Path) -> list:
    """Charge le catalogue d'articles à suivre depuis stock_items.json."""

    if not path.exists():
        raise FileNotFoundError(f"stock_items.json introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    enabled = [i for i in items if i.get("enabled", True)]
    log.info(f"Catalogue : {len(enabled)}/{len(items)} article(s) actif(s) chargé(s)")

    return enabled


def load_stock_state(path: Path) -> dict:
    """Charge l'état de stock depuis stock_state.json. Retourne un dict indexé par sku."""

    if not path.exists():
        log.warning(f"stock_state.json introuvable : {path} - état de stock vide")

        return {}
    with open(path, "r", encoding="utf-8") as f:
        states = json.load(f)
    result = {s["sku"]: s for s in states if "sku" in s}
    log.info(f"État de stock : {len(result)} SKU(s) chargé(s)")

    return result


# ─── 3. RÉCUPÉRATION ET ENRICHISSEMENT DES TRANSACTIONS ──────────────────────

def fetch_transactions(start: str, end: str, mock_file: str = None) -> list:
    if mock_file:
        log.info(f" [MOCK] Lecture depuis '{mock_file}'")
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
    log.info(f"Total brut récupéré : {len(items)} transaction(s)")

    return items


def enrich_transactions(txns: list, headers: dict) -> list:
    log.info(f"Enrichissement de {len(txns)} transaction(s)…")
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
                log.warning(f"↳ {txn_id} : réponse {resp.status_code}")
        except Exception as e:
            log.warning(f"↳ Échec enrichissement {txn_id} : {e}")
        enriched.append(t)
        time.sleep(0.1)
    log.info(f"Enrichissement terminé : {len(enriched)} transaction(s)")

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
        return sku_index[key]["sku"], sku_index[key]

    # Correspondance name seul si la config n'a pas de variant
    key_no_variant = (norm_name, "")

    if key_no_variant in sku_index:
        return sku_index[key_no_variant]["sku"], sku_index[key_no_variant]

    # Correspondance partielle : le nom SumUp contient le nom config

    for (idx_name, idx_variant), item in sku_index.items():
        if idx_name and idx_name in norm_name:
            if not idx_variant or idx_variant in norm_variant:
                return item["sku"], item

    return None, None


# ─── 5. AGRÉGATION HEBDOMADAIRE ──────────────────────────────────────────────

def aggregate_weekly_sales(
    txns: list,
    sku_index: dict,
    weeks_range: list,
) -> tuple:
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
            sku, item = match_product_to_sku(summary, "", sku_index)

            if sku:
                weekly_sales[sku][week_label] += 1

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

            sku, item = match_product_to_sku(name, variant, sku_index)

            if sku:
                pack = item.get("pack_size", 1) or 1
                weekly_sales[sku][week_label] += qty * pack
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
    weekly_demand = max(float(avg_rolling4 or 0), 0.0)
    lead_time_days = int(item.get("supplier_lead_time_days", 7) or 7)
    lead_time_weeks = max(1, math.ceil(lead_time_days / 7))

    if weekly_demand <= 0:
        return {
            "weekly_demand": 0.0,
            "lead_time_weeks": lead_time_weeks,
            "safety_stock": 0,
            "reorder_point": 0,
            "target_stock": 0,
        }

    safety_stock = math.ceil(max(weekly_demand, sales_7d))
    reorder_point = math.ceil((weekly_demand * lead_time_weeks) + safety_stock)
    target_stock = math.ceil(weekly_demand * max(3, lead_time_weeks + 2))

    if target_stock < reorder_point:
        target_stock = reorder_point

    return {
        "weekly_demand": round(weekly_demand, 2),
        "lead_time_weeks": lead_time_weeks,
        "safety_stock": int(safety_stock),
        "reorder_point": int(reorder_point),
        "target_stock": int(target_stock),
    }


def compute_indicators(item: dict, state: dict, weekly_sales: dict, weeks_range: list) -> dict:
    sku = item["sku"]
    sales_by_week = weekly_sales.get(sku, {})

    sales_series = [sales_by_week.get(w, 0) for w in weeks_range]
    total_sold = sum(sales_series)
    n_weeks = len(weeks_range)
    n_zero_weeks = sum(1 for s in sales_series if s == 0)

    sales_7d = sales_series[-1] if sales_series else 0
    sales_28d = sum(sales_series[-4:]) if len(sales_series) >= 4 else sum(sales_series)

    avg_weekly = total_sold / n_weeks if n_weeks > 0 else 0
    last4 = sales_series[-4:] if len(sales_series) >= 4 else sales_series
    avg_rolling4 = sum(last4) / len(last4) if last4 else 0

    prev_week_sales = sales_series[-2] if len(sales_series) >= 2 else None
    variation_pct = None

    if prev_week_sales is not None and prev_week_sales > 0:
        variation_pct = ((sales_7d - prev_week_sales) / prev_week_sales) * 100
    elif prev_week_sales == 0 and sales_7d > 0:
        variation_pct = 100.0

    proj_next_week = round(avg_rolling4, 1)
    proj_4_weeks = round(avg_rolling4 * PROJECTION_WEEKS, 1)

    stock_on_hand = int(state.get("stock_on_hand", 0) or 0)
    stock_reserved = int(state.get("stock_reserved", 0) or 0)
    incoming_qty = int(state.get("incoming_qty", 0) or 0)
    incoming_eta = state.get("incoming_eta") or None
    last_inventory_date = state.get("last_inventory_date") or "N/A"
    inventory_method = state.get("inventory_count_method") or "N/A"

    available_stock = stock_on_hand - stock_reserved

    thresholds = compute_dynamic_thresholds(item, avg_rolling4, sales_7d)
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
        rupture_dt = date.today() + timedelta(weeks=coverage_weeks)
        rupture_date = rupture_dt.isoformat()

    qty_to_order = max(0, target_stock - effective_stock_now)

    if avg_rolling4 <= 0 and effective_stock_now <= 0:
        status = "N/A"
    elif avg_rolling4 > 0 and effective_stock_now <= 0:
        status = "RISQUE RUPTURE"
    elif avg_rolling4 > 0 and coverage_weeks is not None and coverage_weeks < lead_time_weeks:
        status = "RISQUE RUPTURE"
    elif effective_stock_now <= reorder_point:
        status = "A COMMANDER"
    elif effective_stock_now <= max(safety_stock, math.ceil(reorder_point * 1.15)):
        status = "SURVEILLANCE"
    else:
        status = "OK"

    return {
        "sku": sku,
        "label": item.get("label", sku),
        "category": item.get("category", ""),
        "unit": item.get("unit", "piece"),
        "sumup_match": item.get("sumup_match", {}),

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

        "sales_series": sales_series,
        "weeks_range": weeks_range,
        "total_sold": total_sold,
        "sales_7d": sales_7d,
        "sales_28d": sales_28d,
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
    "accent": (60, 120, 220),
    "text_dark": (40, 42, 48),
    "text_mid": (120, 124, 135),
    "text_light": (170, 173, 182),
    "row_even": (246, 247, 250),
    "row_odd": (255, 255, 255),
    "divider": (210, 213, 220),
    "OK": (30, 115, 70),
    "SURVEILLANCE": (180, 130, 20),
    "A COMMANDER": (200, 80, 20),
    "RISQUE RUPTURE": (160, 38, 58),
    "N/A": (150, 150, 150),
    "status_bg": {
        "OK": (220, 245, 230),
        "SURVEILLANCE": (255, 245, 220),
        "A COMMANDER": (255, 228, 210),
        "RISQUE RUPTURE": (255, 215, 220),
        "N/A": (240, 240, 240),
    },
}


class StockPDF(FPDF):
    def __init__(self, week_label: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.week_label = week_label
        self.set_margins(14, 8, 14)
        self.set_auto_page_break(True, margin=16)

    def _pw(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def _safe(self, text, max_len=999) -> str:
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

    def _status_color(self, status: str) -> tuple:
        return PALETTE.get(status, PALETTE["N/A"])

    def header(self):
        self.set_font("Helvetica", "", 8)
        pw = self._pw()
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
        self.set_y(-13)
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_light"])
        self.cell(0, 10, f"SumUp - Rapport Stocks | Page {self.page_no()}", align="C")

    # ── Titre de section ────────────────────────────────────────────────────
    def section_title(self, title: str, color: tuple = None):
        color = color or PALETTE["accent"]
        self.ln(2)
        self.set_fill_color(*color)
        y = self.get_y()
        self.rect(self.l_margin, y, 3, 7, style="F")
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(
            self._pw() - 5,
            7,
            self._safe(title.upper()),
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
        pw = self._pw()
        col_w = pw / 2

        for i, (label, value) in enumerate(kpis):
            if i % 2 == 0 and i > 0:
                self.ln(0)
            x_offset = self.l_margin + (col_w * (i % 2))
            self.set_xy(x_offset, self.get_y())
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_w * 0.55, 6, self._safe(label), border=0, align="L")
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*PALETTE["text_dark"])
            self.cell(col_w * 0.45, 6, self._safe(str(value)), border=0,
                      align="L", new_x="RIGHT" if i % 2 == 0 else "LMARGIN",
                      new_y="TOP" if i % 2 == 0 else "NEXT")

        if len(kpis) % 2 != 0:
            self.ln(6)
        self.ln(2)

    # ── Badge statut ────────────────────────────────────────────────────────
    def status_badge(self, status: str):
        color = self._status_color(status)
        bg = PALETTE["status_bg"].get(status, (240, 240, 240))
        self.set_fill_color(*bg)
        self.set_draw_color(*color)
        self.set_line_width(0.4)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        w = min(60, self._pw())
        self.cell(w, 8, f"  {status}  ", border=1, fill=True,
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_line_width(0.2)
        self.set_draw_color(*PALETTE["divider"])
        self.set_text_color(*PALETTE["text_dark"])

    # ── Tableau des ventes hebdomadaires ────────────────────────────────────
    def weekly_table(self, kpi: dict):
        weeks = kpi["weeks_range"]
        sales = kpi["sales_series"]
        pw = self._pw()
        n = len(weeks)

        if n == 0:
            return

        col_week = pw * 0.30
        col_qty = pw * 0.20
        col_avg = pw * 0.25
        col_var = pw * 0.25
        row_h = 6.0
        head_h = 7.0

        # Calcul des moyennes glissantes et variations
        rows = []

        for i, (w, s) in enumerate(zip(weeks, sales)):
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
            rows.append((w, s, avg, var_str))

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
        self.cell(col_qty, head_h, "Qte vendue", border=0, align="R")
        self.cell(col_avg, head_h, "Moy. glissante", border=0, align="R")
        self.cell(col_var, head_h, "Variation", border=0, align="R",
                  new_x="LMARGIN", new_y="NEXT")
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

        # Lignes

        for i, (w, s, avg, var_str) in enumerate(rows):
            if self.get_y() + row_h > self.h - self.b_margin:
                self.add_page()

            if i % 2 == 0:
                self.set_fill_color(*PALETTE["row_even"])
                self.rect(self.l_margin, self.get_y(), pw, row_h, style="F")
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_week, row_h, w, border="B", align="C")
            self.set_text_color(*PALETTE["text_dark"])
            self.cell(col_qty, row_h, str(s), border="B", align="R")
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(col_avg, row_h, f"{avg:.1f}", border="B", align="R")
            # Couleur variation
            try:
                var_val = float(var_str.replace("+", "").replace("%", ""))

                if var_val > 0:
                    self.set_text_color(30, 115, 70)
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
        import io
        import matplotlib.pyplot as plt

        weeks = kpi["weeks_range"]
        sales = kpi["sales_series"]

        if not weeks or not sales:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*PALETTE["text_mid"])
            self.cell(
                0, 6,
                self._safe("Aucune donnée hebdomadaire disponible."),
                new_x="LMARGIN", new_y="NEXT"
            )
            self.set_text_color(*PALETTE["text_dark"])

            return

        current_stock = float(kpi["available_stock"])
        incoming_qty = float(kpi["incoming_qty"])
        avg_week = float(kpi["avg_rolling4"] or 0)

        stock_curve = []
        running_stock = current_stock + incoming_qty

        for qty in reversed(sales):
            running_stock += qty

        for qty in sales:
            running_stock -= qty
            stock_curve.append(max(running_stock, 0))

        future_weeks = []
        future_stock = []
        trend_stock = stock_curve[-1] if stock_curve else current_stock

        last_label = weeks[-1]
        try:
            year = int(last_label.split("-W")[0])
            week = int(last_label.split("-W")[1])
        except Exception:
            year = datetime.now().year
            week = 1

        for i in range(1, 5):
            next_week = week + i
            next_year = year

            while next_week > 52:
                next_week -= 52
                next_year += 1
            future_weeks.append(f"{next_year}-W{next_week:02d}")
            trend_stock -= avg_week
            future_stock.append(max(trend_stock, 0))

        all_labels = weeks + future_weeks
        all_stock = stock_curve + future_stock

        rupture_index = None

        for i, val in enumerate(all_stock):
            if val <= 0:
                rupture_index = i

                break

        fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=160)

        ax.plot(range(len(weeks)), stock_curve, color="#3c78dc", linewidth=2.2,
                marker="o", markersize=4, label="Stock estime")

        ax.plot(
            range(len(weeks) - 1, len(all_labels)),
            [stock_curve[-1]] + future_stock if stock_curve else future_stock,
            color="#cc4125",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=3.5,
            label="Tendance"
        )

        ax.axhspan(0, max(1, kpi["safety_stock"]), color="#f4cccc", alpha=0.35, label="Zone basse")
        ax.axhline(kpi["safety_stock"], color="#e69138", linestyle=":", linewidth=1.3)
        ax.axhline(kpi["reorder_point"], color="#bf9000", linestyle=":", linewidth=1.3)

        if rupture_index is not None:
            ax.axvline(rupture_index, color="#990000", linestyle="--", linewidth=1.2)
            label_y = max(all_stock) * 0.18 if max(all_stock) > 0 else 1
            ax.annotate(
                f"Rupture estimee\n{all_labels[rupture_index]}",
                xy=(rupture_index, max(all_stock[rupture_index], 0)),
                xytext=(rupture_index + 0.2, label_y),
                fontsize=8,
                color="#990000",
                arrowprops=dict(arrowstyle="->", color="#990000", lw=1),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#990000", alpha=0.9),
            )

        for i, y in enumerate(stock_curve):
            ax.annotate(str(int(round(y))), (i, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color="#3c78dc")

        for i, y in enumerate(future_stock, start=len(weeks)):
            ax.annotate(str(int(round(y))), (i, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color="#cc4125")

        ax.set_title("Evolution du stock et tendance", fontsize=11)
        ax.set_ylabel("Quantite")
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=35, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

        ymax = max(
            [1]
            + [float(v) for v in stock_curve]
            + [float(v) for v in future_stock]
            + [float(kpi["target_stock"] or 0)]
            + [float(kpi["reorder_point"] or 0)]
        )
        ax.set_ylim(0, ymax * 1.20)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)

        chart_w = self._pw()
        chart_h = 72
        y0 = self.get_y()

        if y0 + chart_h > self.h - self.b_margin:
            self.add_page()
            y0 = self.get_y()

        self.image(buf, x=self.l_margin, y=y0, w=chart_w, h=chart_h)
        self.set_y(y0 + chart_h + 4)

# ─── Page 1 : Synthèse globale ────────────────────────────────────────────────


def render_page_summary(pdf: StockPDF, all_kpis: list, week_label: str, weeks_range: list):
    pdf.add_page()
    pw = pdf._pw()

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
    col_lbl = pw * 0.42
    col_stk = pw * 0.12
    col_cov = pw * 0.12
    col_cmd = pw * 0.12
    col_sta = pw * 0.12
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
        status_color = pdf._status_color(status)
        cov = f"{kpi['coverage_weeks']:.1f} sem." if kpi['coverage_weeks'] is not None else "N/A"

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*PALETTE["text_mid"])
        # pdf.cell(col_sku, row_h, pdf._safe(kpi["sku"], 20), border="B", align="L")
        pdf.set_text_color(*PALETTE["text_dark"])
        pdf.cell(col_lbl, row_h, pdf._safe(kpi["label"], 40), border="B", align="L")
        pdf.cell(col_stk, row_h, str(kpi["available_stock"]), border="B", align="R")
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
    pdf.add_page()
    pw = pdf._pw()

    # ── En-tête article ──
    sm = kpi["sumup_match"]
    variant_str = sm.get("variant") or "(sans variante)"
    pdf.section_title(f"Article : {kpi['label']}  [{kpi['sku']}]")

    pdf.kpi_block([
        # ("SKU", kpi["sku"]),
        ("Libelle", kpi["label"]),
        ("Variante SumUp", variant_str),
        # ("Categorie", kpi["category"]),
        ("Unite", kpi["unit"]),
        ("Stock disponible", kpi["available_stock"]),
        ("Stock arrivant", kpi["incoming_qty"]),
        ("ETA reappro", kpi["incoming_eta"] or "N/A"),
        ("Stock securite (auto)", kpi["safety_stock"]),
        ("Point de commande (auto)", kpi["reorder_point"]),
        ("Stock cible (auto)", kpi["target_stock"]),
        ("Dernier inventaire", kpi["last_inventory_date"]),
    ])

    # Badge statut
    pdf.status_badge(kpi["status"])

    # ── Bloc KPIs ──
    pdf.section_title("Indicateurs cles")
    cov = f"{kpi['coverage_weeks']:.1f} sem." if kpi["coverage_weeks"] is not None else "N/A"
    # var = f"{kpi['variation_pct']:+.1f}%" if kpi["variation_pct"] is not None else "N/A"
    pdf.kpi_block([
        # ("Ventes 7 jours", kpi["sales_7d"]),
        ("Ventes 28 jours", kpi["sales_28d"]),
        ("Moyenne hebdomadaire simu", kpi["avg_weekly"]),
        ("Moy. glissante 4 sem.", kpi["avg_rolling4"]),
        # ("Projection sem. suiv.", kpi["proj_next_week"]),
        # ("Projection vente 4 sem.", kpi["proj_4_weeks"]),
        ("Couverture estimee", cov),
        ("Date rupture estimee", kpi["rupture_date"] or "N/A"),
        (f"Qte a commander [{kpi["unit"]}]", kpi["qty_to_order"]),
        # ("Variation S vs S-1", var),
        # ("Sem. sans vente", kpi["n_zero_weeks"]),
        ("Total vendu (periode)", kpi["total_sold"]),
    ])

    # ── Tableau hebdomadaire ──
    pdf.section_title("Evolution du stock")
    pdf.weekly_graph(kpi)
    pdf.weekly_table(kpi)


# ─── Dernière page : qualité des données ─────────────────────────────────────

def render_data_quality_page(pdf: StockPDF, unmapped: list, all_kpis: list):
    pdf.add_page()
    pdf.section_title("Qualite des donnees", PALETTE["text_mid"])

    pw = pdf._pw()

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
            pdf.cell(col_name, 5.5, pdf._safe(u["name"], 40), border="B", align="L")
            pdf.set_text_color(*PALETTE["text_mid"])
            pdf.cell(col_var, 5.5, pdf._safe(u["variant"], 30), border="B", align="L")
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
        pdf.cell(pw * 0.55, 5.5, pdf._safe(kpi["label"], 40), border=0, align="L")
        pdf.set_text_color(*PALETTE["text_dark"])
        pdf.cell(pw * 0.30, 5.5, str(kpi["last_inventory_date"]), border=0, align="L")
        pdf.cell(pw * 0.15, 5.5, kpi["inventory_method"], border=0, align="L",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(*PALETTE["text_dark"])


# ─── Génération complète du PDF ───────────────────────────────────────────────

def generate_pdf(all_kpis: list, unmapped: list, week_label: str, weeks_range: list,
                 path: str) -> str:
    pdf = StockPDF(week_label)
    render_page_summary(pdf, all_kpis, week_label, weeks_range)

    for kpi in all_kpis:
        render_article_page(pdf, kpi)
    render_data_quality_page(pdf, unmapped, all_kpis)
    pdf.output(path)
    log.info(f"PDF généré -> {path}")

    return path


# ─── 8. EXPORTS CSV ───────────────────────────────────────────────────────────

def export_csv_summary(all_kpis: list, path: str):
    fields = [
        # "sku", "label", "category", "unit",
        "available_stock", "incoming_qty", "incoming_eta",
        "safety_stock", "reorder_point", "target_stock",
        "sales_7d", "sales_28d", "avg_weekly", "avg_rolling4",
        "proj_next_week", "proj_4_weeks",
        "coverage_weeks", "rupture_date", "qty_to_order",
        "variation_pct", "n_zero_weeks", "total_sold",
        "status", "last_inventory_date", "inventory_method",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_kpis)
    log.info(f"CSV synthèse -> {path}")


def export_csv_history(all_kpis: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sku", "label", "week", "qty_sold"])

        for kpi in all_kpis:
            for w, s in zip(kpi["weeks_range"], kpi["sales_series"]):
                writer.writerow([kpi["sku"], kpi["label"], w, s])
    log.info(f"CSV historique -> {path}")


# ─── 9. ENVOI EMAIL ───────────────────────────────────────────────────────────

def send_stock_email(
    pdf_path: str,
    csv_path: str,
    week_label: str,
    all_kpis: list,
):
    n_alert = sum(1 for k in all_kpis if k["status"] in ("SURVEILLANCE", "A COMMANDER", "RISQUE RUPTURE"))
    to_order = [k for k in all_kpis if k["status"] in ("A COMMANDER", "RISQUE RUPTURE")]

    order_lines = ""

    if to_order:
        order_lines = "\nArticles a commander :\n"

        for k in to_order:
            order_lines += (
                f"  - {k['label']} [{k['sku']}] : {k['qty_to_order']} {k['unit']}(s)  "
                f"- Statut : {k['status']}\n"
            )
    else:
        order_lines = "\nAucun article a commander cette semaine.\n"

    logs_str = build_log_footer(_log_buffer)
    now_str = datetime.now().strftime("%d/%m/%Y a %H:%M")

    body = f"""\
Bonjour,

Veuillez trouver en piece jointe le rapport hebdomadaire de gestion des stocks
pour la semaine {week_label}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUME - Semaine {week_label}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Articles suivis     : {len(all_kpis)}
Articles en alerte  : {n_alert}
{order_lines}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOURNAL D'EXECUTION - genere le {now_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{logs_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cordialement,
Corentin via sumup_stocks.py
"""
    subject = (
        f"Rapport Stocks SumUp - {week_label} "
        f"({len(all_kpis)} articles, {n_alert} alerte(s))"
    )
    attachments = [pdf_path]
    if csv_path and Path(csv_path).exists():
        attachments.append(csv_path)

    send_email(
        subject=subject,
        body=body,
        attachments=attachments,
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
    items_file = items_file or BASE_DIR / "stock_items.json"
    state_file = state_file or BASE_DIR / "stock_state.json"

    now = datetime.now(timezone.utc)
    # Calcul de la fenêtre temporelle
    end_dt = now
    start_dt = end_dt - timedelta(weeks=weeks)
    start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Semaines ISO dans la plage
    weeks_range = []
    cursor = start_dt
    seen = set()
    while cursor <= end_dt:
        lbl = iso_week_label(cursor)
        if lbl not in seen:
            weeks_range.append(lbl)
            seen.add(lbl)
        cursor += timedelta(days=7)
    weeks_range = sorted(set(weeks_range))

    current_week = iso_week_label(now)
    log.info(f"══ Rapport Stocks SumUp ══ {weeks} semaines | Semaine courante : {current_week}")

    # 1. Chargement configuration
    log.info("Étape 1/5 - Chargement des fichiers de configuration…")
    stock_items = load_stock_items(items_file)
    stock_state = load_stock_state(state_file)
    sku_index = build_sku_index(stock_items)

    # 2. Transactions
    log.info("Étape 2/5 - Récupération des transactions…")
    headers_api = {"Authorization": f"Bearer {SUMUP_API_KEY}"}
    all_txns = fetch_transactions(start, end, mock_file=mock_file)
    if not mock_file:
        all_txns = enrich_transactions(all_txns, headers_api)

    # 3. Agrégation
    log.info("Étape 3/5 - Agrégation hebdomadaire…")
    weekly_sales, unmapped = aggregate_weekly_sales(all_txns, sku_index, weeks_range)
    if unmapped:
        log.warning(f"{len(unmapped)} produit(s) SumUp non mappé(s) au catalogue")

    # 4. Calcul indicateurs
    log.info("Étape 4/5 - Calcul des indicateurs…")
    all_kpis = []
    for item in stock_items:
        state = stock_state.get(item["sku"], {})
        kpi = compute_indicators(item, state, weekly_sales, weeks_range)
        all_kpis.append(kpi)
        log.info(
            f"  {kpi['sku']:30s} | stock={kpi['available_stock']:4d} "
            f"| vendu={kpi['total_sold']:4d} | statut={kpi['status']}"
        )

    # 5. Génération des fichiers
    log.info("Étape 5/5 - Génération des fichiers…")
    safe_week = current_week.replace("-", "_")
    pdf_path = str(BASE_DIR / f"rapport_stocks_{safe_week}.pdf")
    csv_path = str(BASE_DIR / f"rapport_stocks_{safe_week}.csv")
    hist_path = str(BASE_DIR / f"rapport_stocks_history_{safe_week}.csv")

    generate_pdf(all_kpis, unmapped, current_week, weeks_range, pdf_path)
    export_csv_summary(all_kpis, csv_path)
    export_csv_history(all_kpis, hist_path)

    if send_mail:
        send_stock_email(pdf_path, csv_path, current_week, all_kpis)
    else:
        log.info("Envoi email ignoré (--no-mail).")

    log.info("══ Terminé ══")
    return all_kpis, unmapped


# ─── 11. CLI ──────────────────────────────────────────────────────────────────

def main():
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
        mock_file=args.mock,
        items_file=Path(args.items) if args.items else None,
        state_file=Path(args.state) if args.state else None,
    )


if __name__ == "__main__":
    main()
