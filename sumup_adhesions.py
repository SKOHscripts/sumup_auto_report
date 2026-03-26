#!/usr/bin/env python3
"""
SumUp - Rapport des Adhésions
─────────────────────────────
Usage :
  python sumup_adhesions.py                      # 14 derniers jours
  python sumup_adhesions.py --start 2026-01-01 --end 2026-03-31
  python sumup_adhesions.py --no-mail            # PDF local, sans envoi email

Automatisation via crontab (exemple : chaque lundi à 09:00) :
  0 9 * * 1 /usr/bin/python3 /chemin/vers/sumup_adhesions.py >> /var/log/sumup.log 2>&1
"""

import io
import os
import sys
import argparse
import logging
import smtplib
import ssl
import json
import unicodedata
import time
import math
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from fpdf import FPDF
import requests
from pathlib import Path

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
TRANSACTION_FILTERS = []
DEFAULT_DAYS = 14

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
log = logging.getLogger(__name__)


# ── Capture des logs en mémoire pour l'email ──
_log_buffer = io.StringIO()
_log_handler = logging.StreamHandler(_log_buffer)
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.getLogger().addHandler(_log_handler)

# ─── CHARGEMENT DES CREDENTIALS ──────────────────────────────────────────────
# Fichier .env protégé par chmod 600 (lisible uniquement par votre utilisateur)

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def _load_dotenv():
    if not ENV_FILE.exists():
        raise RuntimeError(f".env introuvable : {ENV_FILE}")
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        log.info(f".env chargé depuis : {Path(ENV_FILE).name}")

    except ImportError:
        raise RuntimeError("python-dotenv non installé : pip install python-dotenv")

    if not os.getenv("SUMUP_API_KEY"):
        raise RuntimeError("SUMUP_API_KEY manquante dans .env")


_load_dotenv()

# ── Toutes les variables depuis .env ----──────────────────────────────────────
SUMUP_API_KEY = os.getenv("SUMUP_API_KEY")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO_LIST = [e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()]


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
                timeout=10
            )

            if resp.status_code == 200:
                detail = resp.json()
                t = {**t, **{k: v for k, v in detail.items() if v is not None}}
            else:
                log.warning(f"  ↳ {txn_id} : réponse {resp.status_code}")
        except Exception as e:
            log.warning(f"  ↳ Échec enrichissement {txn_id} : {e}")

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
            merged: dict = {}

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


def fetch_transactions(start: str, end: str, mock_file: str = None) -> list:
    """GET /v0.1/transactions  - ou lecture depuis un fichier mock."""

    # ── Mode mock ──

    if mock_file:
        log.info(f"  [MOCK] Lecture depuis '{mock_file}'")
        with open(mock_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return data.get("items", data.get("transactions", []))

    # ── Mode réel ──

    if not SUMUP_API_KEY:
        log.error("SUMUP_API_KEY manquante. Ajoutez-la via keyring ou .env.")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {SUMUP_API_KEY}"}
    all_txns, limit, offset = [], 1000, 0

    while True:
        resp = requests.get(
            "https://api.sumup.com/v0.1/me/transactions/history",
            headers=headers,
            params={
                "limit": limit,
                "order": "descending",
                "oldest_time": start,
                "newest_time": end,
                # "statuses[]": ["SUCCESSFUL", "CANCELLED", "FAILED", "REFUNDED", "CHARGE_BACK"],  # ajoute CANCELLED pour les remises 100%
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("items", data.get("transactions", []))
        else:
            items = []

        all_txns.extend(items)
        log.info(f"  ↳ {len(items)} transaction(s) reçue(s) (offset={offset})")

        if len(items) < limit:
            break
        offset += limit

    log.info(f"Total brut récupéré : {len(all_txns)} transaction(s)")

    return all_txns


# ─── 2. FILTRE "Adhésion" ─────────────────────────────────────────────────────

def remove_accents(text: str) -> str:
    """Supprime les accents d'une chaîne (ex: 'é' -> 'e')."""

    if not text:
        return ""
    # NFD décompose les caractères accentués (é -> e + ´), on filtre les diacritiques

    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def filter_adhesions(txns: list, filters: list = None) -> list:
    active = [remove_accents(kw.lower()) for kw in (filters or TRANSACTION_FILTERS) if kw.strip()]

    if not active:
        log.info(f"Aucun filtre actif — {len(txns)} transaction(s) conservée(s)")

        return txns

    filtered = [
        t for t in txns

        if any(
            kw in remove_accents(_get_description(t).lower())

            for kw in active
        )
    ]

    log.info(
        f"Filtre(s) actif(s) : {active} — "
        f"{len(filtered)}/{len(txns)} transaction(s) conservée(s)"
    )

    return filtered


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
    elif "MASTERCARD" in ctype or "MAESTRO" in ctype:
        return "MASTERCARD"
    else:
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
    "CASH": "Especes",
    "VISA": "Visa - Debit",
    "MASTERCARD": "Mastercard - Debit",
    "OTHER": "Autres types de paiement",
}

SECTION_ORDER = ["CASH", "VISA", "MASTERCARD", "OTHER"]

COL_W = {
    "date": 42,
    "desc": 75,
    "amount": 28,
    "currency": 18,
    "card4": 30,
    "status": 28,
    "code": 0,
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
        t = str(text or "-").encode("latin-1", errors="replace").decode("latin-1")

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
        self.cell(pw * 0.58, 9, "  Rapport des Adhesions  SumUp",
                  border=0, fill=False, new_x="RIGHT", new_y="TOP")

        # Période + date à droite
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PALETTE["text_mid"])
        gen = datetime.now().strftime("%d/%m/%Y  %H:%M")
        self.cell(0, 9,
                  f"Periode : {self.start_date}  -  {self.end_date}     Genere le {gen}   ",
                  border=0, fill=False, align="R", new_x="LMARGIN", new_y="NEXT")

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
        self.cell(0, 10,
                  f"SumUp  -  Rapport Adhesions  |  Page {self.page_no()}",
                  align="C")

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
        self.cell(self._pw() - 5, 8,
                  SECTION_LABELS[cat].upper(),
                  border=0, fill=False, new_x="RIGHT", new_y="TOP")

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
        self.set_line_width(0.4)
        self.set_draw_color(*PALETTE["text_mid"])
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_line_width(0.2)
        self.set_draw_color(*PALETTE["divider"])

        self.cell(COL_W["date"], HEAD_H, "Date & Heure", border=0, fill=False, align="C")
        self.cell(COL_W["desc"], HEAD_H, "  Description", border=0, fill=False, align="L")
        self.cell(COL_W["amount"], HEAD_H, "Montant EUR  ", border=0, fill=False, align="R")
        self.cell(COL_W["currency"], HEAD_H, "Devise", border=0, fill=False, align="C")
        self.cell(COL_W["card4"], HEAD_H, "Carte", border=0, fill=False, align="C")
        self.cell(COL_W["status"], HEAD_H, "Statut", border=0, fill=False, align="C")
        self.cell(COL_W["code"], HEAD_H, "Code transaction", border=0, fill=False, align="C", new_x="LMARGIN", new_y="NEXT")

        # Ligne basse de l'entête
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

    # ── Ligne de transaction ─────────────────────────────────────────────────
    def _row_height(self, desc: str) -> float:
        """Calcule la hauteur d'une ligne selon le contenu de la description."""
        self.set_font("Helvetica", "", 7.5)
        n_lines = max(1, math.ceil(self.get_string_width(f"  {desc}") / (COL_W["desc"] - 4)))

        return n_lines * ROW_H

    def _check_page_break(self, needed_h: float):
        """Force un saut de page si la hauteur requise ne tient pas."""

        if self.get_y() + needed_h > self.h - self.b_margin:
            self.add_page()

    def transaction_row(self, txn: dict, even: bool, cat: str):
        # ── Données ──
        raw = txn.get("timestamp", txn.get("transaction_date", ""))
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            date_str = dt.strftime("%d/%m/%Y  %H:%M")
        except Exception:
            date_str = raw[:16] if raw else "-"

        desc = self._safe(_get_description(txn))
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

        # ── Zébrage ──

        if even:
            self.set_fill_color(*PALETTE["row_even"])
            self.rect(x0, y0, self._pw(), dyn_h, style="F")

        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.15)

        # ── Largeur dynamique de la colonne code ──
        code_w = self._pw() - sum(v for k, v in COL_W.items() if k != "code")

        # ── Dessin cellule par cellule avec set_xy absolu ──
        cx = x0

        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        self.set_xy(cx, y0)
        self.cell(COL_W["date"], dyn_h, date_str, border="B", fill=False, align="C")
        cx += COL_W["date"]

        # Description multi-ligne
        self.set_text_color(*PALETTE["text_dark"])
        self.set_xy(cx + 2, y0)
        self.set_auto_page_break(False)           # désactivé le temps du multi_cell
        self.multi_cell(COL_W["desc"] - 2, ROW_H, desc, border=0, fill=False, align="L")
        self.set_auto_page_break(True, margin=16)  # réactivé immédiatement après
        self.line(cx, y0 + dyn_h, cx + COL_W["desc"], y0 + dyn_h)
        cx += COL_W["desc"]

        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*PALETTE["text_dark"])
        self.set_xy(cx, y0)
        self.cell(COL_W["amount"], dyn_h, f"{amount:.2f}  ", border="B", fill=False, align="R")
        cx += COL_W["amount"]

        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        self.set_xy(cx, y0)
        self.cell(COL_W["currency"], dyn_h, currency, border="B", fill=False, align="C")
        cx += COL_W["currency"]

        self.set_xy(cx, y0)
        self.cell(COL_W["card4"], dyn_h, last4_str, border="B", fill=False, align="C")
        cx += COL_W["card4"]

        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*status_color)
        self.set_xy(cx, y0)
        self.cell(COL_W["status"], dyn_h, status_str, border="B", fill=False, align="C")
        cx += COL_W["status"]

        self.set_font("Helvetica", "", 7)
        self.set_text_color(*PALETTE["text_light"])
        self.set_xy(cx, y0)
        self.cell(code_w, dyn_h, code, border="B", fill=False, align="C")

        # ── Repositionne proprement sous la ligne ──
        self.set_xy(x0, y0 + dyn_h)
        self.set_text_color(*PALETTE["text_dark"])

    # ── Sous-total de section ─────────────────────────────────────────────────
    def section_total(self, txns: list, cat: str):
        total = sum(float(t.get("amount", 0) or 0) for t in txns)
        color = PALETTE[cat]
        pw = self._pw()

        self.ln(1)
        # Texte gauche : count, en gris
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*PALETTE["text_mid"])
        self.cell(pw - 68, 6,
                  f"   {len(txns)} transaction(s)  -  {SECTION_LABELS[cat]}",
                  border=0, fill=False, align="L")

        # Texte droit : sous-total coloré
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(68, 6,
                  f"Sous-total : {total:.2f} EUR   ",
                  border=0, fill=False, align="R", new_x="LMARGIN", new_y="NEXT")

        # Ligne de séparation légère
        self.set_draw_color(*PALETTE["divider"])
        self.set_line_width(0.25)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

        self.set_text_color(*PALETTE["text_dark"])
        self.ln(6)


def generate_pdf(groups: dict, start: str, end: str, path: str):
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
            pdf.transaction_row(txn, even=(i % 2 == 0), cat=cat)

        pdf.section_total(txns, cat)

        grand_count += len(txns)
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

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*PALETTE["text_mid"])
    pdf.cell(pw - 80, 11,
             f"   TOTAL GENERAL  -  {grand_count} adhesion(s)",
             border=0, fill=False, align="L")

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*PALETTE["accent"])
    pdf.cell(80, 11,
             f"{grand_total:.2f} EUR   ",
             border=0, fill=False, align="R", new_x="LMARGIN", new_y="NEXT")

    # Ligne basse
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)

    pdf.output(path)
    log.info(f"PDF genere -> {path}  ({grand_count} adhesion(s), {grand_total:.2f} EUR)")

    return grand_count, grand_total
# ─── 5. ENVOI EMAIL (multi-destinataires) ─────────────────────────────────────


def send_email(pdf_path: str, start: str, end: str,
               groups: dict, grand_count: int, grand_total: float):

    if not EMAIL_TO_LIST:
        log.warning("Aucun destinataire défini (EMAIL_TO vide). Email non envoyé.")

        return

    if not all([SMTP_USER, SMTP_PASS]):
        log.warning("SMTP_USER ou SMTP_PASS manquant. Email non envoyé.")

        return

    # ── Résumé par section ──
    section_lines = []

    for cat in SECTION_ORDER:
        txns = groups.get(cat, [])

        if not txns:
            continue
        total = sum(float(t.get("amount", 0) or 0) for t in txns)
        section_lines.append(
            f"  {SECTION_LABELS[cat]:<28} {len(txns):>3} transaction(s)   {total:>8.2f} EUR"
        )
    sections_str = "\n".join(section_lines) if section_lines else "  (aucune transaction)"

    # ── Logs complets ──
    logs_str = _log_buffer.getvalue().strip()

    # ── Corps du mail ──
    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    body = f"""\
Bonjour,

Veuillez trouver en pièce jointe le rapport des adhésions SumUp
pour la période du {start} au {end}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RÉSUMÉ DE LA PÉRIODE  —  {start} au {end}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{sections_str}

  {'─' * 50}
  TOTAL                          {grand_count:>3} transaction(s)   {grand_total:>8.2f} EUR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  JOURNAL D'EXÉCUTION  —  généré le {now_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{logs_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cordialement,
Corentin via {Path(__file__).name}
"""

    em = EmailMessage()
    em["From"] = EMAIL_FROM
    em["To"] = ", ".join(EMAIL_TO_LIST)
    em["Subject"] = (
        f"Rapport Adhésions SumUp — {start} au {end} "
        f"({grand_count} tx, {grand_total:.2f} EUR)"
    )
    em.set_content(body)

    with open(pdf_path, "rb") as f:
        em.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(pdf_path)
        )

    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as srv:
        srv.ehlo()
        srv.starttls(context=ctx)
        srv.ehlo()
        srv.login(SMTP_USER, SMTP_PASS)
        srv.send_message(em)

    log.info(f"Email envoyé à : {', '.join(EMAIL_TO_LIST)}")


# ─── 6. PIPELINE PRINCIPAL ────────────────────────────────────────────────────

def run_report(start: str = None, end: str = None, send_mail: bool = True, mock_file: str = None, filters: list = None):
    now = datetime.now(timezone.utc)

    if not end:
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    if not start:
        end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        start = (end_dt - timedelta(days=DEFAULT_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info(f"══ Rapport SumUp ══  {start[:10]} -> {end[:10]}")

    log.info("Étape 1/4 - Récupération des transactions…")
    headers = {"Authorization": f"Bearer {SUMUP_API_KEY}"}
    all_txns = fetch_transactions(start, end, mock_file=mock_file)

    # Enrichissement uniquement en mode réel (pas en mock)

    if not mock_file:
        all_txns = enrich_transactions(all_txns, headers)

    log.info(f"Étape 2/4 — Filtrage {filters or TRANSACTION_FILTERS}…")
    adhesions = filter_adhesions(all_txns, filters=filters)

    if not adhesions:
        log.warning(f"Aucune adhésion trouvée entre {start[:10]} et {end[:10]}.")

        return

    log.info("Étape 3/4 - Tri et génération du PDF…")
    groups = group_by_payment(adhesions)
    OUTPUT_PDF = f'rapport_adhesions_{start[:10]}_{end[:10]}.pdf'
    grand_count, grand_total = generate_pdf(groups, start[:10], end[:10], OUTPUT_PDF)

    if send_mail:
        log.info("Étape 4/4 - Envoi par email…")
        send_email(OUTPUT_PDF, start[:10], end[:10], groups, grand_count, grand_total)
    else:
        log.info("Étape 4/4 - Envoi email ignoré (--no-mail).")

    log.info("══ Terminé ══")


# ─── 7. CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rapport des Adhésions SumUp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
        )
    parser.add_argument("--start", help="Date début YYYY-MM-DD (défaut : -14 jours)")
    parser.add_argument("--end", help="Date fin   YYYY-MM-DD (défaut : aujourd'hui)")
    parser.add_argument("--no-mail", action="store_true",
                        help="Génère le PDF sans envoyer l'email")
    parser.add_argument("--mock", metavar="FICHIER", help="Utilise un fichier JSON local à la place de l'API (ex: mock_transactions.json)")
    parser.add_argument(
        "--filtres",
        nargs="*",                    # 0 ou plusieurs valeurs
        default=None,
        metavar="MOT",
        help="Mots-clés à filtrer (ex: --filtres Adhesion Don). "
        "Sans argument : utilise TRANSACTION_FILTERS du script. "
        "Avec --filtres sans valeur : toutes les transactions."
        )

    args = parser.parse_args()

    def fmt(d: str) -> str:
        return f"{d}T00:00:00Z" if d else None

    run_report(
        start=fmt(args.start),
        end=fmt(args.end),
        send_mail=not args.no_mail,
        mock_file=args.mock,
        filters=args.filtres,
        )


if __name__ == "__main__":
    main()
