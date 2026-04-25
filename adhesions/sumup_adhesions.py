#!/usr/bin/env python3
"""
SumUp - Rapport des Adhésions
─────────────────────────────
Usage :
python sumup_adhesions.py                      # 14 derniers jours
python sumup_adhesions.py --no-mail            # PDF local, sans envoi email
python sumup_adhesions.py --start 2026-01-01 --end 2026-03-31
python sumup_adhesions.py --mock mock_transactions.json

Automatisation via crontab (exemple : chaque lundi à 09:00) :
0 9 * * 1 /usr/bin/python3 /chemin/vers/sumup_adhesions.py >> /var/log/sumup.log 2>&1
"""

import sys
from pathlib import Path
# Permet l'exécution directe `python adhesions/sumup_adhesions.py` en plus de `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpdf import FPDF
import fpdf as _fpdf
import requests
import re
from datetime import datetime, timedelta, timezone
import unicodedata
import time
import os
import math
import logging
import json
import argparse
from utils.mail_utils import (
    load_project_env,
    setup_memory_log_capture,
    send_email,
    build_log_footer,
    )


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

TRANSACTION_FILTERS = []
DEFAULT_DAYS = 7

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_project_env(
    env_file=ENV_FILE,
    required_vars=["SUMUP_API_KEY"],
    logger=log,
    )

_log_buffer, _log_handler = setup_memory_log_capture()
SUMUP_API_KEY = os.getenv("SUMUP_API_KEY")

# ─── 1. RÉCUPÉRATION API ──────────────────────────────────────────────────────


def enrich_transactions(txns: list, headers: dict) -> list:
    """Récupère le détail complet de chaque transaction via GET /v0.1/me/transactions?id={id}."""
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


def _get_description(txn: dict) -> str:
    """
    Construit la description la plus précise possible :
    - Cas panier  : agrège quantity × Nom (variante) depuis txn["products"]
    - Cas simple  : product_summary
    - Fallbacks   : description, note, "-"
    """
    products = txn.get("products") or []

    if products and isinstance(products, list):
        parts = []

        for p in products:
            if not isinstance(p, dict):
                continue

            name = (p.get("name") or "").strip()
            variant = (p.get("description") or "").strip()
            qty = int(p.get("quantity") or 1)

            # Construit le libellé : "Nom (Variante)" ou juste "Nom"
            label = f"{name} ({variant})" if variant else name

            if label:
                parts.append((label, qty))

        if parts:
            # Regroupe les lignes identiques en additionnant les quantités
            merged = {}

            for label, qty in parts:
                merged[label] = merged.get(label, 0) + qty

            return ", ".join(
                f"{qty}x {label}" if qty > 1 else label

                for label, qty in merged.items()
                )

    # Fallbacks

    if txn.get("product_summary"):
        return txn["product_summary"]

    return txn.get("description") or txn.get("note") or "-"


def _normalize_for_match(text: str) -> str:
    return remove_accents((text or "").lower()).strip()


def _get_active_filters(filters: list = None) -> list:
    active = [remove_accents(kw.lower()) for kw in (filters or TRANSACTION_FILTERS) if kw.strip()]

    if active:
        return active

    return ["adhesion"]


def _matches_adhesion_label(text: str, filters: list = None) -> bool:
    normalized = _normalize_for_match(text)

    return any(kw in normalized for kw in _get_active_filters(filters))


def count_adhesions_in_txn(txn: dict, filters: list = None) -> int:
    products = txn.get("products") or []
    total = 0

    if isinstance(products, list) and products:
        for p in products:
            if not isinstance(p, dict):
                continue

            name = (p.get("name") or "").strip()
            variant = (p.get("description") or "").strip()
            label = f"{name} ({variant})" if variant else name

            if not _matches_adhesion_label(label, filters):
                continue

            try:
                qty = int(p.get("quantity") or 1)
            except Exception:
                qty = 1

            total += max(qty, 1)

        if total > 0:
            return total

    desc = _get_description(txn)

    if not _matches_adhesion_label(desc, filters):
        return 0

    normalized_desc = _normalize_for_match(desc)

    matches = re.findall(r"(\d+)\s*x\s*adhesion", normalized_desc)

    if matches:
        return sum(int(m) for m in matches)

    return 1


def count_adhesions_in_group(txns: list, filters: list = None) -> int:
    return sum(count_adhesions_in_txn(txn, filters=filters) for txn in txns)


def fetch_transactions(start: str, end: str, mock_file: str = None) -> list:
    """GET /v0.1/transactions  - ou lecture depuis un fichier mock."""

    # ── Mode mock ──

    if mock_file:
        log.info(f"  [MOCK] Lecture depuis '{mock_file}'")
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
            "limit": 1000,
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


def remove_accents(text: str) -> str:
    """Supprime les accents d'une chaîne (ex: 'é' -> 'e')."""

    if not text:
        return ""
    # NFD décompose les caractères accentués (é -> e + ´), on filtre les diacritiques

    return "".join(
        c for c in unicodedata.normalize("NFD", text)

        if unicodedata.category(c) != "Mn"
        )


def filter_adhesions(txns: list, filters: list = None) -> list:
    active = [remove_accents(kw.lower()) for kw in (filters or TRANSACTION_FILTERS) if kw.strip()]

    if not active:
        log.info(f"Aucun filtre actif — {len(txns)} transaction(s) conservée(s)")

        return txns

    filtered = [
        t for t in txns

        if any(kw in remove_accents(_get_description(t).lower()) for kw in active)
        ]

    log.info(
        f"Filtre(s) actif(s) : {active} — "
        f"{len(filtered)}/{len(txns)} transaction(s) conservée(s)"
        )

    return filtered


def get_count_label(filters: list = None, default_label: str = "adhésion") -> str:
    active = [kw.strip() for kw in (filters or TRANSACTION_FILTERS) if kw and kw.strip()]

    if not active:
        return default_label

    if len(active) == 1:
        return active[0]

    return " / ".join(active)


def format_count_label(count: int, filters: list = None, default_label: str = "adhésion") -> str:
    label = get_count_label(filters=filters, default_label=default_label)

    if count <= 1:
        return label

    if label.endswith("s"):
        return label

    return f"{label}s"

# ─── 3. CATÉGORISATION PAR TYPE DE PAIEMENT ───────────────────────────────────
# SumUp expose :
#   txn["payment_type"]       -> "CASH", "POS", "ECOM", "MOTO"…
#   txn["card"]["type"]       -> "VISA", "MASTERCARD", "AMEX"…
#   txn["card"]["card_type"]  -> parfois "debit", "credit"


def get_category(txn: dict) -> str:
    """Retourne CASH | VISA | MASTERCARD | OTHER."""
    ptype = (txn.get("payment_type") or "").upper()

    if ptype == "CASH":
        return "CASH"

    # Pour les paiements par carte, lire le sous-objet card
    card = txn.get("card") or {}
    ctype = (card.get("type") or "").upper()

    if "VISA" in ctype:
        return "VISA"

    if "MASTERCARD" in ctype or "MAESTRO" in ctype:
        return "MASTERCARD"

    return "OTHER"


def group_by_payment(txns: list) -> dict:
    """Groupe et trie : CASH en premier, VISA, MASTERCARD, OTHER."""
    groups = {"CASH": [], "VISA": [], "MASTERCARD": [], "OTHER": []}

    for t in txns:
        groups[get_category(t)].append(t)

    for g in groups.values():
        g.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    log.info(
        f"Répartition -> Espèces: {len(groups['CASH'])} | "
        f"Visa: {len(groups['VISA'])} | "
        f"Mastercard: {len(groups['MASTERCARD'])} | "
        f"Autres: {len(groups['OTHER'])}"
        )

    return groups


# ─── 4. GÉNÉRATION PDF ─── Style sobre / print-friendly ──────────────────────
PALETTE = {
    "CASH": (34, 110, 90),    # teal
    "VISA": (50, 75, 165),    # indigo
    "MASTERCARD": (165, 38, 58),    # bordeaux
    "OTHER": (100, 95, 85),    # gris chaud
    "accent": (60, 120, 220),   # bleu acier (header + total)
    "text_dark": (40, 42, 48),     # ardoise foncé (remplace le noir pur)
    "text_mid": (120, 124, 135),  # gris moyen
    "text_light": (170, 173, 182),  # gris clair (pied de page)
    "row_even": (246, 247, 250),  # gris quasi blanc (zébrage léger)
    "row_odd": (255, 255, 255),  # blanc
    "divider": (210, 213, 220),  # séparateur gris clair
    "status": {
        "SUCCESSFUL": (30, 115, 70),
        "FAILED": (160, 38, 58),
        "CANCELLED": (150, 95, 20),
        "PENDING": (70, 75, 170),
        },
    }

SECTION_LABELS = {
    "CASH": "Espèces",
    "VISA": "Visa - Débit",
    "MASTERCARD": "Mastercard - Débit",
    "OTHER": "Autres types de paiement",
    }

SECTION_ORDER = ["CASH", "VISA", "MASTERCARD", "OTHER"]

COL_W = {
    "date": 42,
    "desc": 92,
    "amount": 28,
    "currency": 18,
    "card4": 26,
    "status": 28,
    "code": 42,
    }

ROW_H = 6.5
HEAD_H = 7.5


class AdhesionPDF(FPDF):
    def __init__(self, start_date, end_date):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.start_date = start_date
        self.end_date = end_date
        self.set_margins(14, 8, 14)
        self.set_auto_page_break(True, margin=16)

    def _pw(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def _safe(self, text, max_len=999) -> str:
        t = str(text or "-").replace("€", "EUR")
        t = t.encode("latin-1", errors="replace").decode("latin-1")

        return (t[:max_len - 3] + "...") if len(t) > max_len else t

    # ── En-tête de page ──────────────────────────────────────────────────────
    def header(self):
        # Police obligatoire en premier dans fpdf2 avant tout appel cell()
        self.set_font("Helvetica", "", 8)

        pw = self._pw()

        # Bande accent fine en haut (3mm)
        self.set_fill_color(*PALETTE["accent"])
        self.set_draw_color(*PALETTE["accent"])
        self.cell(0, 3, "", fill=True, border=0, new_x="LMARGIN", new_y="NEXT")

        # Titre
        self.ln(3)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*PALETTE["text_dark"])
        self.cell(pw * 0.58, 9, " Rapport des Adhésions SumUp",
                  border=0, fill=False, new_x="RIGHT", new_y="TOP")

        # Période + date à droite
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PALETTE["text_mid"])
        gen = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.cell(
            0, 9,
            f"Periode : {self.start_date} - {self.end_date}  Genere le {gen}",
            border=0, fill=False, align="R", new_x="LMARGIN", new_y="NEXT"
            )

        # Ligne séparatrice
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    # ── Pied de page ─────────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_light"])
        self.cell(0, 10, f"SumUp - Rapport Adhesions | Page {self.page_no()}", align="C")

    # ── Titre de section ─────────────────────────────────────────────────────
    def section_header(self, cat: str):
        color = PALETTE[cat]
        x, y = self.get_x(), self.get_y()

        # Barre latérale colorée fine (3mm)
        self.set_fill_color(*color)
        self.rect(x, y, 3, 8, style="F")

        # Texte sur fond blanc
        self.set_x(x + 5)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*color)
        self.cell(
            self._pw() - 5, 8,
            SECTION_LABELS[cat].upper(),
            border=0, fill=False, new_x="RIGHT", new_y="TOP"
            )

        # Ligne horizontale à droite du titre
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.2)
        line_y = y + 4
        self.line(self.l_margin + self._pw() * 0.28,
                  line_y,
                  self.w - self.r_margin,
                  line_y)

        self.ln(8)
        self.set_x(self.l_margin)
        self.set_text_color(*PALETTE["text_dark"])

    # ── En-tête du tableau ───────────────────────────────────────────────────
    def table_header(self):
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.2)

        # Ligne haute du tableau
        y = self.get_y()
        self.set_line_width(0.4)
        self.set_draw_color(*PALETTE["text_mid"])
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_line_width(0.2)
        self.set_draw_color(*PALETTE["divider"])

        self.cell(COL_W["date"], HEAD_H, "Date & Heure", border=0, align="C")
        self.cell(COL_W["desc"], HEAD_H, " Description", border=0, align="L")
        self.cell(COL_W["amount"], HEAD_H, "Montant EUR", border=0, align="R")
        self.cell(COL_W["currency"], HEAD_H, "Devise", border=0, align="C")
        self.cell(COL_W["card4"], HEAD_H, "Carte", border=0, align="C")
        self.cell(COL_W["status"], HEAD_H, "Statut", border=0, align="C")
        self.cell(COL_W["code"], HEAD_H, "Code transaction", border=0, align="C", new_x="LMARGIN", new_y="NEXT")

        # Ligne basse de l'entête
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

    # ── Ligne de transaction ─────────────────────────────────────────────────
    def _row_height(self, desc: str) -> float:
        """Calcule la hauteur d'une ligne selon le contenu de la description."""
        self.set_font("Helvetica", "", 7.5)
        usable = max(10, COL_W["desc"] - 6)
        n_lines = max(1, math.ceil(self.get_string_width(desc) / usable))

        return n_lines * ROW_H

    def transaction_row(self, txn: dict, even: bool):
        raw = txn.get("timestamp", txn.get("transaction_date", ""))
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            date_str = dt.strftime("%d/%m/%Y %H:%M")
        except Exception:
            date_str = raw[:16] if raw else "-"

        desc = self._safe(_get_description(txn), max_len=120)
        amount = float(txn.get("amount", 0) or 0)
        currency = self._safe(txn.get("currency", "EUR"), 6)

        card = txn.get("card") or {}
        last4 = card.get("last_4_digits", "")
        last4_str = f"**** {last4}" if last4 else "-"

        status_raw = (txn.get("status") or "").upper()
        status_map = {
            "SUCCESSFUL": "Reussi",
            "FAILED": "Echoue",
            "CANCELLED": "Annule",
            "PENDING": "En attente",
            }
        status_str = status_map.get(status_raw, "-")
        status_color = PALETTE["status"].get(status_raw, PALETTE["text_mid"])
        code = self._safe(str(txn.get("transaction_code", txn.get("id", "-"))), max_len=24)

        # ── Hauteur dynamique ──
        dyn_h = self._row_height(desc)

        # ── Saut de page manuel AVANT de dessiner ──
        # Si la ligne ne tient pas + réaffiche l'entête de tableau sur la nouvelle page

        if self.get_y() + dyn_h > self.h - self.b_margin:
            self.add_page()
            self.table_header()

        x0, y0 = self.l_margin, self.get_y()

        if even:
            self.set_fill_color(*PALETTE["row_even"])
            self.rect(x0, y0, self._pw(), dyn_h, style="F")

        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_mid"])

        cx = x0
        self.set_xy(cx, y0)
        self.cell(COL_W["date"], dyn_h, date_str, border="B", align="C")
        cx += COL_W["date"]

        # Description multi-ligne
        self.set_text_color(*PALETTE["text_dark"])
        self.set_xy(cx + 2, y0)
        self.set_auto_page_break(False)  # désactivé le temps du multi-cell
        self.multi_cell(COL_W["desc"] - 2, ROW_H, desc, border=0, align="L")
        self.set_auto_page_break(True, margin=16)  # réactivé juste après
        self.line(cx, y0 + dyn_h, cx + COL_W["desc"], y0 + dyn_h)
        cx += COL_W["desc"]

        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*PALETTE["text_dark"])
        self.set_xy(cx, y0)
        self.cell(COL_W["amount"], dyn_h, f"{amount:.2f}", border="B", align="R")
        cx += COL_W["amount"]

        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        self.set_xy(cx, y0)
        self.cell(COL_W["currency"], dyn_h, currency, border="B", align="C")
        cx += COL_W["currency"]

        self.set_xy(cx, y0)
        self.cell(COL_W["card4"], dyn_h, last4_str, border="B", align="C")
        cx += COL_W["card4"]

        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*status_color)
        self.set_xy(cx, y0)
        self.cell(COL_W["status"], dyn_h, status_str, border="B", align="C")
        cx += COL_W["status"]

        self.set_font("Helvetica", "", 7)
        self.set_text_color(*PALETTE["text_light"])
        self.set_xy(cx, y0)
        # ── Largeur dynamique de la colonne code ──
        code_w = self._pw() - sum(v for k, v in COL_W.items() if k != "code")
        self.cell(code_w, dyn_h, code, border="B", align="C")

        # ── Repositionne proprement sous la ligne ──
        self.set_xy(x0, y0 + dyn_h)
        self.set_text_color(*PALETTE["text_dark"])

    # ── Sous-total de section ─────────────────────────────────────────────────
    def section_total(self, txns: list, cat: str, filters: list = None):
        total_amount = sum(float(t.get("amount", 0) or 0) for t in txns)
        total_count = count_adhesions_in_group(txns, filters=filters)
        count_label = format_count_label(total_count, filters=filters, default_label="adhésion")
        color = PALETTE[cat]
        pw = self._pw()

        self.ln(1)
        # Texte gauche : count, en gris
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        self.cell(
            pw - 68,
            6,
            f" {total_count} {count_label} - {SECTION_LABELS[cat]}",
            border=0,
            fill=False,
            align="L",
            )

        # Texte droit : sous-total coloré
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(
            68,
            6,
            f"Sous-total : {total_amount:.2f} EUR",
            border=0,
            fill=False,
            align="R",
            new_x="LMARGIN",
            new_y="NEXT",
            )

        # Ligne de séparation légère
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.25)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

        self.set_text_color(*PALETTE["text_dark"])
        self.ln(6)


def generate_pdf(groups: dict, start: str, end: str, path: str, filters: list = None):
    pdf = AdhesionPDF(start, end)
    pdf.add_page()

    grand_count, grand_total = 0, 0.0

    for cat in SECTION_ORDER:
        txns = groups.get(cat, [])

        if not txns:
            continue

        pdf.section_header(cat)
        pdf.table_header()

        for i, txn in enumerate(txns):
            pdf.transaction_row(txn, even=(i % 2 == 0))

        pdf.section_total(txns, cat, filters=filters)

        grand_count += count_adhesions_in_group(txns, filters=filters)
        grand_total += sum(float(t.get("amount", 0) or 0) for t in txns)

    # ── Total général ─────────────────────────────────────────────────────────
    pdf.ln(2)

    # Encadré sobre : juste une bordure accent
    pw = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_draw_color(*PALETTE["accent"])
    pdf.set_line_width(0.5)

    # Ligne haute
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)

    count_label = format_count_label(grand_count, filters=filters, default_label="adhésion")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*PALETTE["text_mid"])
    pdf.cell(
        pw - 80,
        11,
        f" TOTAL GENERAL - {grand_count} {count_label}",
        border=0,
        fill=False,
        align="L",
    )

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*PALETTE["accent"])
    pdf.cell(
        80,
        11,
        f"{grand_total:.2f} EUR",
        border=0,
        fill=False,
        align="R",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # Ligne basse
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)

    pdf.output(path)
    log.info(f"PDF genere -> {path} ({grand_count} {count_label}, {grand_total:.2f} EUR)")

    return grand_count, grand_total

# ─── 5. ENVOI EMAIL (multi-destinataires) ─────────────────────────────────────


def send_report_email(pdf_path: str, start: str, end: str,
                      groups: dict, grand_count: int, grand_total: float,
                      filters: list = None):

    section_lines = []

    for cat in SECTION_ORDER:
        txns = groups.get(cat, [])

        if not txns:
            continue

        total_amount = sum(float(t.get("amount", 0) or 0) for t in txns)
        total_count = count_adhesions_in_group(txns, filters=filters)
        count_label = format_count_label(total_count, filters=filters, default_label="adhésion")

        section_lines.append(
            f" {SECTION_LABELS[cat]:<28} {total_count:>3} {count_label:<12} {total_amount:>8.2f} EUR"
            )

    sections_str = "\n".join(section_lines) if section_lines else " (aucune adhésion)"
    logs_str = build_log_footer(_log_buffer)
    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    total_label = format_count_label(grand_count, filters=filters, default_label="adhésion")

    body = f"""\
Bonjour,

Veuillez trouver en pièce jointe le rapport SumUp
pour la période du {start} au {end}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÉSUMÉ DE LA PÉRIODE — {start} au {end}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{sections_str}

{'─' * 50}
TOTAL {grand_count:>3} {total_label} {grand_total:>8.2f} EUR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOURNAL D'EXÉCUTION — généré le {now_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{logs_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cordialement,
Corentin via {Path(__file__).name}
    """

    subject = (
        f"Rapport SumUp — {start} au {end} "
        f"({grand_count} {total_label}, {grand_total:.2f} EUR)"
        )

    send_email(
        subject=subject,
        body=body,
        attachments=[pdf_path],
        mailing_list="finance",
        logger=log,
        )

# ─── 6. PIPELINE PRINCIPAL ────────────────────────────────────────────────────


def run_report(start: str = None, end: str = None, send_mail: bool = True,
               mock_file: str = None, filters: list = None):
    now = datetime.now(timezone.utc)

    if not end:
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    if not start:
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        start = (end_dt - timedelta(days=DEFAULT_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info(f"══ Rapport SumUp ══ {start[:10]} -> {end[:10]}")
    log.info("Étape 1/4 - Récupération des transactions…")
    headers = {"Authorization": f"Bearer {SUMUP_API_KEY}"}
    all_txns = fetch_transactions(start, end, mock_file=mock_file)

    # Enrichissement uniquement en mode réel (pas en mock)

    if not mock_file:
        all_txns = enrich_transactions(all_txns, headers)

    log.info(f"Étape 2/4 - Filtrage {filters or TRANSACTION_FILTERS}…")
    adhesions = filter_adhesions(all_txns, filters=filters)

    if not adhesions:
        log.warning(f"Aucune adhésion trouvée entre {start[:10]} et {end[:10]}.")
        return

    log.info("Étape 3/4 - Tri et génération du PDF…")
    groups = group_by_payment(adhesions)
    output_pdf = BASE_DIR / f"rapport_adhesions_{start[:10]}_{end[:10]}.pdf"
    grand_count, grand_total = generate_pdf(
        groups,
        start[:10],
        end[:10],
        output_pdf,
        filters=filters,
        )

    if send_mail:
        log.info("Étape 4/4 - Envoi par email…")
        send_report_email(
            output_pdf,
            start[:10],
            end[:10],
            groups,
            grand_count,
            grand_total,
            filters=filters,
            )

    else:
        log.info("Étape 4/4 - Envoi email ignoré (--no-mail).")

    log.info("══ Terminé ══")


# ─── 7. CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rapport des Adhésions SumUp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        )
    parser.add_argument("--start", help="Date début YYYY-MM-DD (défaut : -14 jours)")
    parser.add_argument("--end", help="Date fin YYYY-MM-DD (défaut : aujourd'hui)")
    parser.add_argument("--no-mail", action="store_true",
                        help="Génère le PDF sans envoyer l'email")
    parser.add_argument("--mock", metavar="FICHIER",
                        help="Utilise un fichier JSON local à la place de l'API")
    parser.add_argument(
        "--filtres",
        nargs="*",                    # 0 ou plusieurs valeurs
        default=None,
        metavar="MOT",
        help="Mots-clés à filtrer (ex: --filtres Adhesion Don). "
        "Sans argument : utilise TRANSACTION_FILTERS du script. "
        "Avec --filtres sans valeur : toutes les transactions.",
        )
    args = parser.parse_args()

    def fmt_start(d: str) -> str:
        return f"{d}T00:00:00Z" if d else None

    def fmt_end(d: str) -> str:
        return f"{d}T23:59:59Z" if d else None

    run_report(
        start=fmt_start(args.start),
        end=fmt_end(args.end),
        send_mail=not args.no_mail,
        mock_file=args.mock,
        filters=args.filtres,
        )


if __name__ == "__main__":
    main()
