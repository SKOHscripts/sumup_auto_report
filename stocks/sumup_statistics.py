#!/usr/bin/env python3
"""
Module de statistiques SumUp / stock, factorisé pour enrichissement API,
calcul de métriques ventes/articles/catégories/paiements et génération PDF synthétique.

Usage:
  python sumup_statistics.py --weeks 8 --items stock_items.json --pdf stats.pdf
  python sumup_statistics.py --mock mocktransactions.json --items stock_items.json --no-enrich
"""

import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import requests
from fpdf import FPDF

from utils.mail_utils import (
    load_project_env,
    setup_memory_log_capture,
    send_email,
    build_log_footer,
    )
from utils.sumup_shared import normalize, iso_week_label, safe_float, parse_dt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_WEEKS = 8
SUMUP_HISTORY_URL = "https://api.sumup.com/v0.1/me/transactions/history"
SUMUP_TXN_URL = "https://api.sumup.com/v0.1/me/transactions"

PALETTE = {
    "accent": (0, 129, 138),
    "accent2": (0, 102, 112),
    "text": (64, 59, 58),
    "muted": (110, 109, 108),
    "divider": (214, 218, 224),
    "row_even": (237, 248, 249),
    "cash": (0, 129, 138),
    "cb": (200, 134, 10),
}

CATEGORY_COLORS = {
    "bar": "#00818A",
    "soft": "#FFA70B",
    "snacking": "#14B8A6",
    "adhesions": "#C8860A",
    "boissonschaudes": "#006670",
    "dons": "#FFCB4F",
    "autres": "#8a8480",
}

BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "assets" / "logo_village.png"

load_project_env(
    required_vars=["SUMUP_API_KEY"],
    logger=log,
    )

_log_buffer, _log_handler = setup_memory_log_capture()
SUMUP_API_KEY = os.getenv("SUMUP_API_KEY")


@dataclass
class CatalogItem:
    """Article du catalogue SumUp avec ses propriétés de stock et de mappage."""
    stocksku: str
    label: str
    category: str
    unit: str
    enabled: bool
    is_reference: bool
    sumup_name: str
    sumup_variant: str
    consumption_per_sale: float
    sale_price: Optional[float]
    raw: Dict[str, Any]

    @property
    def display_name(self) -> str:
        """Retourne le label ou le SKU si le label est vide."""
        return self.label or self.stocksku


class Catalog:
    """Catalogue d'articles avec index de recherche par nom/variante."""

    def __init__(self, raw_items: List[Dict[str, Any]]):
        """Initialise le catalogue depuis une liste brute d'articles."""
        self.raw_items = raw_items
        self.items = self._prepare_items(raw_items)
        self.sku_index = self._build_sku_index(self.items)
        self.reference_by_sku = self._build_reference_by_sku(self.items)

    @classmethod
    def from_path(cls, path: Path) -> "Catalog":
        """Charge le catalogue depuis un fichier JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Le catalogue JSON doit contenir une liste d'articles")

        return cls(data)

    def _prepare_items(self, raw_items: List[Dict[str, Any]]) -> List[CatalogItem]:
        """Convertit les articles bruts en objets CatalogItem filtrés et normalisés."""
        items: List[CatalogItem] = []

        for raw in raw_items:
            if not raw.get("enabled", True):
                continue
            sm = raw.get("sumup_match", {}) or {}
            price = raw.get("sale_price") or raw.get("saleprice") or raw.get("sellingprice")
            items.append(
                CatalogItem(
                    stocksku=raw.get("stock_sku") or raw.get("sku") or "",
                    label=raw.get("stock_label") or raw.get("label") or raw.get("stock_sku") or "",
                    category=raw.get("category") or "autres",
                    unit=raw.get("stock_unit") or raw.get("unit") or "piece",
                    enabled=bool(raw.get("enabled", True)),
                    is_reference=bool(raw.get("is_stock_reference") or raw.get("stock_state")),
                    sumup_name=sm.get("name") or raw.get("label") or "",
                    sumup_variant=sm.get("variant") or "",
                    consumption_per_sale=safe_float(
                        raw.get("consumption_per_sale", raw.get("pack_size", 1)), 1.0
                    ),
                    sale_price=safe_float(price, 0.0) if price not in (None, "") else None,
                    raw=raw,
                )
            )

        return items

    def _build_sku_index(self, items: Iterable[CatalogItem]) -> Dict[Tuple[str, str], CatalogItem]:
        """Construit l'index (nom_normalisé, variante_normalisée) → CatalogItem."""
        idx = {}

        for item in items:
            idx[(normalize(item.sumup_name), normalize(item.sumup_variant))] = item

        return idx

    def _build_reference_by_sku(self, items: Iterable[CatalogItem]) -> Dict[str, CatalogItem]:
        """Construit l'index SKU → article de référence (is_stock_reference prioritaire)."""
        refs: Dict[str, CatalogItem] = {}

        for item in items:
            if item.stocksku not in refs or item.is_reference:
                refs[item.stocksku] = item

        return refs

    def match_product(self, name: str, variant: str) -> Optional[CatalogItem]:
        """Cherche l'article correspondant au nom+variante, avec fallback partiel."""
        key = (normalize(name), normalize(variant))

        if key in self.sku_index:
            return self.sku_index[key]
        key2 = (normalize(name), "")

        if key2 in self.sku_index:
            return self.sku_index[key2]
        n_name = normalize(name)
        n_variant = normalize(variant)

        for (idx_name, idx_variant), item in self.sku_index.items():
            if idx_name and idx_name in n_name:
                if not idx_variant or idx_variant in n_variant:
                    return item

        return None


class SumUpClient:
    """Client HTTP pour l'API SumUp (transactions et enrichissement)."""

    def __init__(self, api_key: Optional[str], timeout: int = 20):
        """Initialise le client avec la clé API et le timeout."""
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def fetch_transactions(self, start: str, end: str, mock_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les transactions SumUp sur la période ou depuis un fichier mock."""
        if mock_file:
            with open(mock_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data.get("items") or data.get("transactions") or []

            if isinstance(data, list):
                return data

            return []

        if not self.api_key:
            raise RuntimeError("SUMUP_API_KEY manquante et aucun fichier --mock fourni")
        resp = requests.get(
            SUMUP_HISTORY_URL,
            headers=self.headers,
            params={"limit": 5000, "order": "descending", "oldest_time": start, "newest_time": end},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            return data.get("items") or data.get("transactions") or []

        if isinstance(data, list):
            return data

        return []

    def enrich_transactions(
        self, txns: List[Dict[str, Any]], enrich: bool = True, pause: float = 0.08,
    ) -> List[Dict[str, Any]]:
        """Enrichit chaque transaction via GET /v0.1/me/transactions?id=."""
        if not enrich or not self.api_key:
            return txns
        enriched = []

        for t in txns:
            txn_id = t.get("id") or t.get("transaction_id")

            if not txn_id:
                enriched.append(t)

                continue
            try:
                resp = requests.get(SUMUP_TXN_URL, headers=self.headers, params={"id": txn_id}, timeout=10)

                if resp.status_code == 200:
                    detail = resp.json()

                    if isinstance(detail, dict):
                        merged = dict(t)
                        merged.update({k: v for k, v in detail.items() if v is not None})
                        t = merged
                else:
                    log.warning("Enrichissement impossible pour %s (%s)", txn_id, resp.status_code)
            except Exception as e:
                log.warning("Erreur enrichissement %s: %s", txn_id, e)
            enriched.append(t)
            time.sleep(pause)

        return enriched


class TransactionAnalyzer:
    """Analyse et normalise les transactions SumUp selon le catalogue."""

    def __init__(self, catalog: Catalog):
        """Initialise l'analyseur avec le catalogue de référence."""
        self.catalog = catalog

    def extract_products(self, txn: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait la liste de produits d'une transaction."""
        products = txn.get("products") or []
        out = []

        if isinstance(products, list) and products:
            for p in products:
                if not isinstance(p, dict):
                    continue
                qty = int(safe_float(p.get("quantity"), 1))
                name = p.get("name") or ""
                variant = p.get("description") or p.get("variant") or ""
                price = safe_float(p.get("price"), 0.0)
                total_price = safe_float(p.get("total_price"), 0.0) or price * qty
                out.append({
                    "name": name,
                    "variant": variant,
                    "quantity": qty,
                    "unit_price": price if price else None,
                    "line_total": total_price if total_price else None,
                })
        elif txn.get("product_summary"):
            out.append({
                "name": txn.get("product_summary"),
                "variant": "",
                "quantity": 1,
                "unit_price": None,
                "line_total": safe_float(txn.get("amount"), 0.0) or None,
            })

        return out

    def detect_payment_method(self, txn: Dict[str, Any]) -> str:
        """Détecte le moyen de paiement (cash / cb / autre) depuis la transaction."""
        candidates = [
            txn.get("payment_type"),
            txn.get("payment_method"),
            txn.get("card_type"),
            txn.get("transaction_code"),
            txn.get("type"),
        ]
        joined = " ".join(str(x or "") for x in candidates).lower()

        if "cash" in joined or "espe" in joined:
            return "cash"

        return "cb"

    def normalize_transactions(self, txns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalise les transactions en lignes enrichies avec mappage catalogue."""
        rows = []

        for txn in txns:
            status = (txn.get("status") or "").upper()

            if status in {"FAILED", "CANCELLED"}:
                continue
            dt = parse_dt(txn.get("timestamp") or txn.get("transaction_date") or txn.get("local_time"))

            if not dt:
                continue
            week = iso_week_label(dt)
            payment_method = self.detect_payment_method(txn)
            amount = safe_float(txn.get("amount"), 0.0)
            products = self.extract_products(txn)

            if not products:
                continue

            for p in products:
                item = self.catalog.match_product(p["name"], p["variant"])
                sale_price = p["unit_price"] if p.get("unit_price") else (item.sale_price if item else None)
                revenue = (p["line_total"] if p.get("line_total")
                           else (sale_price * p["quantity"] if sale_price is not None else None))
                rows.append({
                    "transaction_id": txn.get("id") or txn.get("transaction_id"),
                    "datetime": dt,
                    "week": week,
                    "payment_method": payment_method,
                    "transaction_amount": amount,
                    "product_name": p["name"],
                    "product_variant": p["variant"],
                    "quantity": p["quantity"],
                    "mapped": item is not None,
                    "stocksku": item.stocksku if item else None,
                    "label": item.display_name if item else p["name"],
                    "category": item.category if item else "autres",
                    "unit": item.unit if item else "piece",
                    "consumption": (item.consumption_per_sale * p["quantity"]) if item else p["quantity"],
                    "unit_sale_price": sale_price,
                    "estimated_revenue": revenue,
                })

        return rows

    def compute_metrics(self, rows: List[Dict[str, Any]], norm_weeks: int = 0) -> Dict[str, Any]:
        """Calcule les métriques agrégées (top articles, catégories, paiements).

        norm_weeks : diviseur de normalisation.
            0 = nombre de semaines actives par article (semaines où l'article a eu des ventes).
            N > 0 = période fixe identique pour tous les articles.
        """
        weeks = sorted({r["week"] for r in rows})
        by_article = defaultdict(
            lambda: {"qty": 0, "revenue": 0.0, "category": "", "mapped": False, "weeks_seen": set()}
        )
        by_category_week = defaultdict(lambda: defaultdict(lambda: {"qty": 0, "revenue": 0.0}))
        by_category = defaultdict(lambda: {"qty": 0, "revenue": 0.0})
        payments = Counter()
        payment_amounts = defaultdict(float)
        unmapped = Counter()

        for r in rows:
            key = r["label"]
            by_article[key]["qty"] += r["quantity"]
            by_article[key]["revenue"] += safe_float(r.get("estimated_revenue"), 0.0)
            by_article[key]["category"] = r["category"]
            by_article[key]["mapped"] = r["mapped"]
            by_article[key]["weeks_seen"].add(r["week"])

            cat = r["category"] or "autres"
            by_category_week[cat][r["week"]]["qty"] += r["quantity"]
            by_category_week[cat][r["week"]]["revenue"] += safe_float(r.get("estimated_revenue"), 0.0)
            by_category[cat]["qty"] += r["quantity"]
            by_category[cat]["revenue"] += safe_float(r.get("estimated_revenue"), 0.0)

            pm = r["payment_method"]
            payments[pm] += 1
            payment_amounts[pm] += safe_float(r.get("estimated_revenue"), 0.0)

            if not r["mapped"]:
                unmapped[(r["product_name"], r["product_variant"])] += r["quantity"]

        for v in by_article.values():
            divisor = norm_weeks if norm_weeks > 0 else len(v["weeks_seen"])
            divisor = divisor or 1
            v["qty_per_week"] = round(v["qty"] / divisor, 2)
            v["revenue_per_week"] = round(v["revenue"] / divisor, 2)
            v["n_active_weeks"] = len(v["weeks_seen"])

        top_articles = sorted(
            [{"label": k, **v} for k, v in by_article.items()],
            key=lambda x: (-x["qty_per_week"], -x["revenue_per_week"], x["label"]),
        )
        top_articles_revenue = sorted(
            [{"label": k, **v} for k, v in by_article.items() if v["revenue"] > 0],
            key=lambda x: (-x["revenue_per_week"], -x["qty_per_week"], x["label"]),
        )
        least_articles = sorted(
            [{"label": k, **v} for k, v in by_article.items()],
            key=lambda x: (x["qty_per_week"], x["revenue_per_week"], x["label"]),
        )

        norm_label = f"{norm_weeks} sem. fixe" if norm_weeks > 0 else "sem. actives par article"

        return {
            "weeks": weeks,
            "rows": rows,
            "norm_weeks": norm_weeks,
            "norm_label": norm_label,
            "top_articles": top_articles,
            "top_articles_revenue": top_articles_revenue,
            "least_articles": least_articles,
            "by_category": dict(by_category),
            "by_category_week": {k: dict(v) for k, v in by_category_week.items()},
            "payment_counts": dict(payments),
            "payment_amounts": dict(payment_amounts),
            "unmapped": [
                {"name": k[0], "variant": k[1], "qty": v}

                for k, v in sorted(unmapped.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            ],
            "total_qty": sum(r["quantity"] for r in rows),
            "total_revenue": round(sum(safe_float(r.get("estimated_revenue"), 0.0) for r in rows), 2),
            "mapped_rows": sum(1 for r in rows if r["mapped"]),
            "total_rows": len(rows),
        }


class ChartFactory:
    """Génère et sauvegarde les graphiques matplotlib pour le rapport PDF."""

    def __init__(self, output_dir: Path):
        """Initialise la factory avec le répertoire de sortie des images."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_weekly_category_qty_chart(self, metrics: Dict[str, Any]) -> Optional[Path]:
        """Génère le graphique de ventes hebdomadaires par catégorie."""
        weeks = metrics["weeks"]
        cats = sorted(metrics["by_category_week"].keys())

        if not weeks or not cats:
            return None
        brand_cycle = ["#00818A", "#FFA70B", "#14B8A6", "#C8860A", "#E05A2B", "#006670", "#FFCB4F"]
        plt.figure(figsize=(10, 4.5), dpi=160)
        plt.gca().set_prop_cycle(color=brand_cycle)

        for cat in cats:
            values = [metrics["by_category_week"].get(cat, {}).get(w, {}).get("qty", 0) for w in weeks]
            plt.plot(weeks, values, marker="o", linewidth=2, label=cat)
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Ventes hebdomadaires par catégorie")
        plt.ylabel("Quantité vendue")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend(fontsize=7, ncol=min(4, max(1, len(cats))))
        plt.tight_layout()
        path = self.output_dir / "weekly_category_qty.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    def save_category_revenue_chart(self, metrics: Dict[str, Any]) -> Optional[Path]:
        """Génère le graphique de chiffre d'affaires estimé par catégorie."""
        data = sorted(metrics["by_category"].items(), key=lambda kv: (-kv[1]["revenue"], kv[0]))

        if not data:
            return None
        cats = [k for k, _ in data]
        vals = [v["revenue"] for _, v in data]
        colors = [CATEGORY_COLORS.get(c, "#7f7f7f") for c in cats]
        plt.figure(figsize=(9, 4.8), dpi=160)
        bars = plt.bar(cats, vals, color=colors)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Chiffre d'affaires estimé par catégorie")
        plt.ylabel("Montant estimé")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        path = self.output_dir / "category_revenue.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    def save_payment_ratio_chart(self, metrics: Dict[str, Any]) -> Optional[Path]:
        """Génère le graphique camembert du ratio cash / CB."""
        counts = metrics["payment_counts"]

        if not counts:
            return None
        labels = []
        values = []
        colors = []

        for k, color in [("cash", "#00818A"), ("cb", "#FFA70B")]:
            if counts.get(k, 0) > 0:
                labels.append(k.upper())
                values.append(counts[k])
                colors.append(color)

        if not values:
            return None
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=160)
        ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors,
               textprops={"fontsize": 10}, pctdistance=0.75, labeldistance=1.15)
        ax.set_title("Ratio ventes cash / CB", pad=12)
        fig.tight_layout(pad=1.5)
        path = self.output_dir / "payment_ratio.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        return path


class StatsPDF(FPDF):
    """PDF du rapport de statistiques : en-tête, pied de page, tableaux et graphiques."""

    def __init__(self, title: str):
        """Initialise le PDF en portrait A4 avec marges et saut de page automatique."""
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title = title
        self.set_auto_page_break(True, 14)
        self.set_margins(12, 10, 12)

    def _safe(self, text: str, max_len: int = 999) -> str:
        """Nettoie le texte pour l'encodage latin-1 et tronque si nécessaire."""
        t = str(text or "-")

        for src, dst in [("€", "EUR"), ("—", "-"), ("–", "-"), ("'", "'"), (" ", " ")]:
            t = t.replace(src, dst)
        t = t.encode("latin-1", errors="replace").decode("latin-1")

        return (t[: max_len - 3] + "...") if len(t) > max_len else t

    def header(self):
        """Affiche le titre et la date de génération en haut de chaque page."""
        logo_h = 14
        logo_w = 22
        y_logo = self.get_y() + 1
        if LOGO_PATH.exists():
            self.image(str(LOGO_PATH), x=self.l_margin, y=y_logo, w=logo_w, h=logo_h)

        text_x = self.l_margin + logo_w + 3
        text_w = self.w - text_x - self.r_margin
        self.set_xy(text_x, y_logo + 2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*PALETTE["text"])
        self.cell(text_w, 6, self._safe(self.title, 90), new_x="LMARGIN", new_y="NEXT")
        self.set_xy(text_x, y_logo + 8)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PALETTE["muted"])
        self.cell(text_w, 5, f"Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")

        self.set_y(y_logo + logo_h + 2)
        self.set_draw_color(*PALETTE["divider"])
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def footer(self):
        """Affiche le numéro de page centré en bas de chaque page."""
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PALETTE["muted"])
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def section(self, title: str):
        """Insère un titre de section avec barre colorée."""
        self.ln(1)
        self.set_fill_color(*PALETTE["accent"])
        self.rect(self.l_margin, self.get_y(), 2.6, 6, style="F")
        self.set_x(self.l_margin + 4)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*PALETTE["accent"])
        self.cell(0, 6, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*PALETTE["text"])
        self.ln(1)

    def kv_table(self, rows: List[Tuple[str, str]]):
        """Affiche un tableau clé/valeur sur deux colonnes avec alternance de fond."""
        left = 58
        right = self.w - self.l_margin - self.r_margin - left

        for i, (k, v) in enumerate(rows):
            if i % 2 == 0:
                self.set_fill_color(*PALETTE["row_even"])
                self.rect(self.l_margin, self.get_y(), left + right, 6.2, style="F")
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*PALETTE["muted"])
            self.cell(left, 6.2, self._safe(k, 40), border="B")
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*PALETTE["text"])
            self.cell(right, 6.2, self._safe(v, 80), border="B", new_x="LMARGIN", new_y="NEXT")

    def simple_table(self, headers: List[str], rows: List[List[str]], widths: List[float]):
        """Affiche un tableau multi-colonnes avec en-têtes et saut de page automatique."""
        self.set_font("Helvetica", "B", 7.8)
        self.set_text_color(*PALETTE["muted"])

        for h, w in zip(headers, widths):
            self.cell(w, 6.5, self._safe(h, 30), border="B", align="L")
        self.ln()

        for i, row in enumerate(rows):
            if self.get_y() > self.h - self.b_margin - 10:
                self.add_page()
                self.set_font("Helvetica", "B", 7.8)
                self.set_text_color(*PALETTE["muted"])

                for h, w in zip(headers, widths):
                    self.cell(w, 6.5, self._safe(h, 30), border="B", align="L")
                self.ln()

            if i % 2 == 0:
                self.set_fill_color(*PALETTE["row_even"])
                self.rect(self.l_margin, self.get_y(), sum(widths), 6.2, style="F")
            self.set_font("Helvetica", "", 7.7)
            self.set_text_color(*PALETTE["text"])

            for cell, w in zip(row, widths):
                self.cell(w, 6.2, self._safe(str(cell), 34), border="B", align="L")
            self.ln()

    def add_chart(self, img_path: Optional[Path], h: float = 62, w: Optional[float] = None):
        """Insère un graphique PNG centré sur la page courante."""
        if not img_path or not img_path.exists():
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*PALETTE["muted"])
            self.cell(0, 5, "Aucun graphique disponible.", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(*PALETTE["text"])

            return

        if self.get_y() + h > self.h - self.b_margin:
            self.add_page()
        page_w = self.w - self.l_margin - self.r_margin
        chart_w = w if w is not None else page_w
        x = self.l_margin + (page_w - chart_w) / 2
        self.image(str(img_path), x=x, y=self.get_y(), w=chart_w, h=h)
        self.ln(h + 2)


class ReportBuilder:
    """Orchestre la génération des graphiques et la création du PDF de rapport."""

    def __init__(self, output_dir: Path):
        """Initialise le builder avec le répertoire de sortie."""
        self.output_dir = output_dir
        self.chart_factory = ChartFactory(output_dir)

    def generate_pdf(self, metrics: Dict[str, Any], pdf_path: Path, title: str):
        """Génère le PDF complet (graphiques + tableaux) à partir des métriques calculées."""
        weekly_chart = self.chart_factory.save_weekly_category_qty_chart(metrics)
        revenue_chart = self.chart_factory.save_category_revenue_chart(metrics)
        payment_chart = self.chart_factory.save_payment_ratio_chart(metrics)

        pdf = StatsPDF(title)
        pdf.add_page()

        payment_counts = metrics["payment_counts"]
        total_payments = sum(payment_counts.values()) or 1
        cash_ratio = payment_counts.get("cash", 0) / total_payments * 100
        cb_ratio = payment_counts.get("cb", 0) / total_payments * 100
        mapped_ratio = metrics["mapped_rows"] / (metrics["total_rows"] or 1) * 100

        pdf.section("Synthèse")
        pdf.kv_table([
            ("Semaines analysées", str(len(metrics["weeks"]))),
            ("Quantité totale vendue", str(metrics["total_qty"])),
            ("CA estimé total", f"{metrics['total_revenue']:.2f} EUR"),
            ("Normalisation", metrics["norm_label"]),
            ("Taux de mapping catalogue", f"{mapped_ratio:.0f} %"),
            ("Ratio cash", f"{cash_ratio:.0f} %"),
            ("Ratio CB", f"{cb_ratio:.0f} %"),
        ])

        pdf.section("Top articles vendus")
        top_qty_rows = [
            [a["label"], a["category"], f"{a['qty_per_week']:.1f}", str(a["qty"]), str(a["n_active_weeks"])]

            for a in metrics["top_articles"][:10]
        ]
        pdf.simple_table(
            ["Article", "Catégorie", "Qté/sem", "Qté tot.", "Sem. act."],
            top_qty_rows, [68, 34, 20, 22, 26],
        )

        pdf.section("Articles les plus rentables")
        top_rev_rows = [
            [a["label"], a["category"], f"{a['revenue_per_week']:.2f}", f"{a['revenue']:.2f}", str(a["n_active_weeks"])]

            for a in metrics["top_articles_revenue"][:10]
        ]
        pdf.simple_table(
            ["Article", "Catégorie", "CA/sem (EUR)", "CA tot. (EUR)", "Sem. act."],
            top_rev_rows, [60, 32, 30, 32, 26],
        )

        pdf.section("Articles les moins vendus")
        low_rows = [
            [a["label"], a["category"], f"{a['qty_per_week']:.1f}", str(a["qty"]), str(a["n_active_weeks"])]

            for a in metrics["least_articles"][:10]
        ]
        pdf.simple_table(
            ["Article", "Catégorie", "Qté/sem", "Qté tot.", "Sem. act."],
            low_rows, [68, 34, 20, 22, 26],
        )

        pdf.add_page()
        pdf.section("Ventes hebdomadaires par catégorie")
        pdf.add_chart(weekly_chart, h=64)

        pdf.section("CA estimé par catégorie")
        cat_rows = [
            [cat, str(int(vals["qty"])), f"{vals['revenue']:.2f} EUR"]

            for cat, vals in sorted(metrics["by_category"].items(), key=lambda kv: (-kv[1]["revenue"], kv[0]))
        ]
        pdf.simple_table(["Catégorie", "Qté", "CA estimé"], cat_rows, [72, 28, 70])
        pdf.add_chart(revenue_chart, h=60)

        pdf.section("Ratio paiements")
        pdf.kv_table([
            ("Ventes cash", str(payment_counts.get("cash", 0))),
            ("Ventes CB", str(payment_counts.get("cb", 0))),
            ("Montant cash estimé", f"{metrics['payment_amounts'].get('cash', 0.0):.2f} EUR"),
            ("Montant CB estimé", f"{metrics['payment_amounts'].get('cb', 0.0):.2f} EUR"),
        ])
        pdf.add_chart(payment_chart, h=74, w=74)

        if metrics["unmapped"]:
            pdf.section("Produits non mappés")
            unm_rows = [[u["name"], u["variant"] or "-", str(u["qty"])] for u in metrics["unmapped"][:12]]
            pdf.simple_table(["Nom", "Variante", "Qté"], unm_rows, [82, 60, 28])

        pdf.output(str(pdf_path))

        return pdf_path


def send_statistics_email(_weeks: int, pdf_path: Path, metrics: Dict[str, Any]) -> None:
    """Compose et envoie l'email de rapport statistiques avec le PDF en pièce jointe."""
    n_weeks = len(metrics["weeks"])
    period_start = metrics["weeks"][0] if metrics["weeks"] else "N/A"
    period_end = metrics["weeks"][-1] if metrics["weeks"] else "N/A"

    total_qty = metrics["total_qty"]
    total_revenue = metrics["total_revenue"]

    payment_counts = metrics["payment_counts"]
    total_payments = sum(payment_counts.values()) or 1
    cash_ratio = payment_counts.get("cash", 0) / total_payments * 100
    cb_ratio = payment_counts.get("cb", 0) / total_payments * 100
    mapped_ratio = metrics["mapped_rows"] / (metrics["total_rows"] or 1) * 100
    n_unmapped = len(metrics["unmapped"])

    norm_label = metrics.get("norm_label", "sem. actives par article")

    top5_qty = metrics["top_articles"][:5]
    top5_qty_str = "\n".join(
        f"  {i + 1}. {a['label']} ({a['category']}) : {a['qty_per_week']:.1f}/sem "
        f"({a['qty']} tot., {a['n_active_weeks']} sem. act.)"

        for i, a in enumerate(top5_qty)
    )

    top5_rev = metrics["top_articles_revenue"][:5]
    top5_rev_str = "\n".join(
        f"  {i + 1}. {a['label']} ({a['category']}) : {a['revenue_per_week']:.2f} EUR/sem "
        f"({a['revenue']:.2f} EUR tot., {a['n_active_weeks']} sem. act.)"

        for i, a in enumerate(top5_rev)
    )

    least5 = metrics["least_articles"][:5]
    least5_str = "\n".join(
        f"  {i + 1}. {a['label']} ({a['category']}) : {a['qty_per_week']:.1f}/sem "
        f"({a['qty']} tot., {a['n_active_weeks']} sem. act.)"

        for i, a in enumerate(least5)
    )

    cats = sorted(metrics["by_category"].items(), key=lambda kv: -kv[1]["revenue"])
    cats_str = "\n".join(
        f"  - {cat:20s} : {int(v['qty']):5d} ventes, {v['revenue']:8.2f} EUR"

        for cat, v in cats
    )

    logs_str = build_log_footer(_log_buffer)
    now_str = datetime.now().strftime("%d/%m/%Y a %H:%M")

    subject = (
        f"Rapport Statistiques SumUp - {period_start} a {period_end} "
        f"({total_qty} ventes, {total_revenue:.0f} EUR)"
    )

    body = f"""\
Bonjour,

Veuillez trouver en piece jointe le rapport de statistiques des ventes SumUp.

======================================================
RESUME -- {period_start} a {period_end} ({n_weeks} semaines)
======================================================
Quantite totale vendue  : {total_qty}
CA estime total         : {total_revenue:.2f} EUR
Normalisation           : {norm_label}
Ratio cash / CB         : {cash_ratio:.0f}% / {cb_ratio:.0f}%
Montant cash estime     : {metrics['payment_amounts'].get('cash', 0.0):.2f} EUR
Montant CB estime       : {metrics['payment_amounts'].get('cb', 0.0):.2f} EUR
Taux de mapping         : {mapped_ratio:.0f}%
Produits non mappes     : {n_unmapped}
======================================================

TOP 5 ARTICLES - QUANTITE VENDUE (normalise) :
{top5_qty_str}

TOP 5 ARTICLES - CA ESTIME (normalise) :
{top5_rev_str}

ARTICLES LES MOINS VENDUS (5 derniers, normalises) :
{least5_str}

VENTES PAR CATEGORIE :
{cats_str}

======================================================
Genere le {now_str}
Cordialement,
Corentin via sumup_statistics.py

--- Logs ---
{logs_str}
"""

    send_email(
        subject=subject,
        body=body,
        attachments=[str(pdf_path)],
        mailing_list="default",
        logger=log,
    )


def run_report(
    weeks: int,
    items_file: Path,
    pdf_path: Path,
    mock_file: Optional[str] = None,
    api_key: Optional[str] = None,
    enrich: bool = True,
    send_mail: bool = True,
    norm_weeks: int = 0,
) -> Path:
    """Exécute le pipeline complet : chargement, transactions, métriques, PDF, email."""
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(weeks=weeks)

    items_file = items_file or BASE_DIR / "stocks" / "stock_items.json"

    log.info("== Rapport Statistiques SumUp == %s semaines", weeks)

    log.info("Etape 1/4 - Chargement du catalogue...")
    catalog = Catalog.from_path(items_file)
    log.info("Catalogue charge : %s article(s) actif(s)", len(catalog.items))

    log.info("Etape 2/4 - Recuperation des transactions...")
    client = SumUpClient(api_key=api_key or SUMUP_API_KEY)
    txns = client.fetch_transactions(
        start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        mock_file=mock_file,
    )
    log.info("Transactions recuperees : %s", len(txns))
    txns = client.enrich_transactions(txns, enrich=enrich)

    log.info("Etape 3/4 - Analyse et calcul des metriques...")
    analyzer = TransactionAnalyzer(catalog)
    rows = analyzer.normalize_transactions(txns)
    metrics = analyzer.compute_metrics(rows, norm_weeks=norm_weeks)
    log.info(
        "Analyse : %s unites vendues, CA estime %.2f EUR, mapping %s/%s lignes",
        metrics['total_qty'], metrics['total_revenue'],
        metrics['mapped_rows'], metrics['total_rows'],
    )
    if metrics["unmapped"]:
        log.warning("%s produit(s) non mappes au catalogue", len(metrics['unmapped']))

    log.info("Etape 4/4 - Generation du PDF...")
    title = f"Rapport statistiques ventes - {weeks} semaines"
    ReportBuilder(pdf_path.parent).generate_pdf(metrics, pdf_path, title)
    log.info("PDF genere -> %s", pdf_path)

    if send_mail:
        send_statistics_email(weeks, pdf_path, metrics)
    else:
        log.info("Envoi email ignore (--no-mail).")

    log.info("== Termine ==")
    return pdf_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit et retourne le parseur d'arguments CLI."""
    p = argparse.ArgumentParser(
        description="Rapport de statistiques des ventes SumUp avec PDF synthetique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--weeks", type=int, default=DEFAULT_WEEKS,
                   help=f"Nombre de semaines analysees (defaut : {DEFAULT_WEEKS})")
    p.add_argument("--items", default=None,
                   help="Chemin vers le catalogue JSON (defaut : stocks/stock_items.json)")
    p.add_argument("--pdf", default=None,
                   help="Chemin du PDF de sortie (defaut : rapport_statistiques_sumup_YYYY_WNN.pdf)")
    p.add_argument("--mock", default=None, help="Fichier JSON de transactions mock")
    p.add_argument("--api-key", default=None, help="Cle API SumUp ; sinon variable d'environnement SUMUP_API_KEY")
    p.add_argument("--no-enrich", action="store_true", help="Desactive l'enrichissement transaction par transaction")
    p.add_argument("--no-mail", action="store_true", help="Genere le PDF sans envoyer l'email")
    p.add_argument(
        "--norm-weeks", type=int, default=0,
        help=(
            "Periode de normalisation en semaines (defaut : 0 = semaines actives par article). "
            "Exemple : --norm-weeks 8 divise par 8 pour tous les articles."
        ),
    )
    return p


def main():
    """Point d'entrée : parse les arguments et lance run_report."""
    args = build_arg_parser().parse_args()
    api_key = args.api_key or SUMUP_API_KEY

    now = datetime.now(timezone.utc)
    week_label = iso_week_label(now).replace("-", "_")
    default_pdf = BASE_DIR / "stocks" / f"rapport_statistiques_sumup_{week_label}.pdf"
    items_file = Path(args.items) if args.items else BASE_DIR / "stocks" / "stock_items.json"

    run_report(
        weeks=args.weeks,
        items_file=items_file,
        pdf_path=Path(args.pdf) if args.pdf else default_pdf,
        mock_file=args.mock or os.getenv("SUMUP_MOCK_FILE") or None,
        api_key=api_key,
        enrich=not args.no_enrich,
        send_mail=not args.no_mail,
        norm_weeks=args.norm_weeks,
    )


if __name__ == "__main__":
    main()
