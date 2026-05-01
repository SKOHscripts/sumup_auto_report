#!/usr/bin/env python3
"""
Mise à jour des stocks depuis le fichier d'achats Google Drive.
──────────────────────────────────────────────────────────────
Télécharge le fichier Excel ACHATS_suivi_stock.xlsx depuis Google Drive,
parse les colonnes d'achat (date + acheteur + quantités), et met à jour
stock_items.json en ajoutant les quantités achetées au stock_on_hand.

Usage :
  python -m stocks.update_stock_from_purchases            # depuis Google Drive
  python -m stocks.update_stock_from_purchases --dry-run  # simulation sans écriture
  python -m stocks.update_stock_from_purchases --local FICHIER.xlsx

Variables d'environnement requises (sauf --local) :
  GDRIVE_SERVICE_ACCOUNT_FILE  Chemin vers le JSON du service account Google
  GDRIVE_PURCHASES_FILE_ID     ID du fichier Excel dans Google Drive
"""

import argparse
import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stocks.gdrive_loader import download_file_as_bytes
from utils.mail_utils import load_project_env
from utils.sumup_shared import normalize

try:
    import openpyxl  # type: ignore[import-untyped]
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False

log = logging.getLogger(__name__)

STOCKS_DIR = Path(__file__).resolve().parent
STOCK_ITEMS_PATH = STOCKS_DIR / "stock_items.json"
PURCHASE_MAPPING_PATH = STOCKS_DIR / "purchase_mapping.json"

# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class PurchaseItem:
    """Un produit acheté dans une colonne du fichier Excel."""

    excel_label: str
    qty: float


@dataclass
class PurchaseEvent:
    """Un achat complet : une colonne du fichier Excel."""

    purchase_date: date
    buyer: str
    items: list[PurchaseItem] = field(default_factory=list)


# ── Lecture / écriture stock_items.json ───────────────────────────────────────

def _load_stock_items_raw(path: Path) -> list:
    """Charge et retourne la liste brute JSON depuis stock_items.json."""
    if not path.exists():
        raise FileNotFoundError(f"stock_items.json introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("stock_items.json doit contenir une liste JSON.")
    return items


def _save_stock_items(path: Path, raw_items: list) -> None:
    """Sauvegarde atomique via fichier temporaire → renommage."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


# ── Parsing du fichier Excel ───────────────────────────────────────────────────

def parse_purchases_excel(excel_bytes: bytes) -> list[PurchaseEvent]:
    """Parse le fichier Excel et retourne la liste des événements d'achat.

    Structure attendue du fichier :
      Ligne 0 : titre
      Ligne 1 : date màj
      Ligne 2 : marqueurs "exemple" (colonnes à ignorer)
      Ligne 3 : prénoms acheteurs  (col C+)
      Ligne 4 : dates d'achat      (col C+)
      Ligne 5 : vide
      Ligne 6+ : données produits par catégorie
        col A : type d'unité (propagé vers le bas)
        col B : nom catégorie (MAJUSCULES) ou nom produit
        col C+: quantités achetées
    """
    if not _OPENPYXL_AVAILABLE:
        raise ImportError(
            "openpyxl est requis pour lire les fichiers Excel. "
            "Installez : pip install openpyxl"
        )

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), data_only=True, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 6:
        raise ValueError("Le fichier Excel ne contient pas assez de lignes.")

    row_example = rows[2]   # ligne 2 : marqueurs "exemple"
    row_buyers  = rows[3]   # ligne 3 : prénoms acheteurs
    row_dates   = rows[4]   # ligne 4 : dates d'achat

    # Identifie les colonnes de données (à partir de la colonne C = index 2)
    # et filtre les colonnes "exemple"
    purchase_cols: list[tuple[int, date, str]] = []  # (col_idx, date, buyer)
    for col_idx in range(2, len(row_dates)):
        if col_idx < len(row_example):
            marker = str(row_example[col_idx] or "").strip().lower()
            if "exemple" in marker:
                continue

        raw_date = row_dates[col_idx] if col_idx < len(row_dates) else None
        if raw_date is None:
            continue

        purchase_date = _parse_excel_date(raw_date)
        if purchase_date is None:
            continue

        # buyer = str(row_buyers[col_idx] or "").strip() if col_idx < len(row_buyers) else ""
        buyer = str("").strip() if col_idx < len(row_buyers) else ""
        purchase_cols.append((col_idx, purchase_date, buyer))

    if not purchase_cols:
        log.warning("Aucune colonne d'achat valide trouvée dans le fichier Excel.")
        return []

    events: dict[int, PurchaseEvent] = {
        col_idx: PurchaseEvent(purchase_date=d, buyer=b)
        for col_idx, d, b in purchase_cols
    }

    for row in rows[6:]:
        col_b = str(row[1] or "").strip() if len(row) > 1 else ""

        if not col_b:
            continue

        # Ignore les en-têtes de catégorie (tout en majuscules)
        if col_b == col_b.upper() and col_b.replace("/", "").replace("&", "").replace(" ", "").isupper():
            continue

        for col_idx, _, _ in purchase_cols:
            raw_qty = row[col_idx] if col_idx < len(row) else None
            if raw_qty is None:
                continue
            try:
                qty = float(raw_qty)
            except (TypeError, ValueError):
                continue
            if qty <= 0:
                continue
            events[col_idx].items.append(PurchaseItem(excel_label=col_b, qty=qty))

    result = [ev for ev in events.values() if ev.items]
    log.info("%d événement(s) d'achat détecté(s) dans le fichier Excel.", len(result))
    return result


def _parse_excel_date(raw_value) -> Optional[date]:
    """Convertit une valeur de cellule Excel en objet date."""
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(raw_value, fmt).date()
            except ValueError:
                continue
    return None


# ── Chargement du mapping ─────────────────────────────────────────────────────

def load_purchase_mapping(path: Path) -> dict[str, tuple[str, float]]:
    """Charge purchase_mapping.json → dict[label_normalisé → (stock_sku, multiplicateur)]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, tuple[str, float]] = {}
    for entry in data.get("products", []):
        label = entry.get("excel_label", "")
        sku = entry.get("stock_sku", "")
        mult = float(entry.get("qty_multiplier", 1))
        if label and sku:
            mapping[normalize(label)] = (sku, mult)

    log.info("%d produit(s) chargé(s) depuis le mapping.", len(mapping))
    return mapping


# ── Déduplication ─────────────────────────────────────────────────────────────

def find_already_processed_dates(raw_items: list) -> set[date]:
    """Retourne les dates déjà intégrées via stock_history (type='purchase')."""
    processed: set[date] = set()
    for item in raw_items:
        state = item.get("stock_state") or {}
        for entry in state.get("stock_history", []):
            if entry.get("type") == "purchase":
                try:
                    processed.add(date.fromisoformat(entry["date"]))
                except (KeyError, ValueError):
                    pass
    return processed


# ── Application des achats ────────────────────────────────────────────────────

def apply_purchases_to_stock(
    raw_items: list,
    events: list[PurchaseEvent],
    mapping: dict[str, tuple[str, float]],
    already_processed: set[date],
    dry_run: bool = False,
) -> tuple[list, list[str], list[str]]:
    """Applique les événements d'achat sur raw_items.

    Pour chaque événement dont la date n'est pas déjà traitée :
      - Résout chaque excel_label en stock_sku via le mapping
      - Ajoute qty * multiplicateur au stock_on_hand du reference item
      - Ajoute une entrée stock_history de type 'purchase'

    Returns:
        (raw_items_mis_à_jour, liste_de_succès, liste_de_warnings)
    """
    ref_items_by_sku: dict[str, dict] = {}
    for item in raw_items:
        if not item.get("enabled", True):
            continue
        sku = item.get("stock_sku") or item.get("sku", "")
        has_state = bool(item.get("stock_state"))
        is_ref = item.get("is_stock_reference", has_state)
        if is_ref and sku and sku not in ref_items_by_sku:
            ref_items_by_sku[sku] = item

    successes: list[str] = []
    warnings: list[str] = []
    today_str = date.today().isoformat()

    for event in sorted(events, key=lambda e: e.purchase_date):
        if event.purchase_date in already_processed:
            log.info("Achat du %s déjà intégré → ignoré.", event.purchase_date.isoformat())
            continue

        date_str = event.purchase_date.isoformat()
        log.info(
            "Traitement achat %s par %s (%d produit(s)).",
            date_str, event.buyer or "?", len(event.items)
        )

        for purchase_item in event.items:
            norm_label = normalize(purchase_item.excel_label)
            if norm_label not in mapping:
                msg = (
                    f"Produit Excel non trouvé dans le mapping : "
                    f"'{purchase_item.excel_label}' (achat du {date_str})"
                )
                warnings.append(msg)
                log.warning(msg)
                continue

            stock_sku, multiplier = mapping[norm_label]

            if stock_sku not in ref_items_by_sku:
                msg = (
                    f"stock_sku '{stock_sku}' introuvable dans stock_items.json "
                    f"(produit : '{purchase_item.excel_label}', achat du {date_str})"
                )
                warnings.append(msg)
                log.warning(msg)
                continue

            qty_to_add = purchase_item.qty * multiplier
            ref_item = ref_items_by_sku[stock_sku]

            if "stock_state" not in ref_item or ref_item["stock_state"] is None:
                ref_item["stock_state"] = {
                    "stock_on_hand": 0,
                    "stock_reserved": 0,
                    "incoming_qty": 0,
                    "incoming_eta": "",
                    "last_inventory_date": today_str,
                    "inventory_count_method": "manual",
                    "stock_history": [],
                }

            state = ref_item["stock_state"]
            prev_stock = float(state.get("stock_on_hand") or 0)
            new_stock = round(prev_stock + qty_to_add, 6)

            history_entry = {
                "type": "purchase",
                "date": date_str,
                "buyer": event.buyer or "",
                "qty_added": qty_to_add,
                "previous_stock_on_hand": prev_stock,
                "new_stock_on_hand": new_stock,
                "source": "gdrive_excel",
            }

            label = ref_item.get("stock_label") or ref_item.get("label") or stock_sku
            unit = ref_item.get("stock_unit") or ref_item.get("unit") or ""
            msg = (
                f"[{date_str}] {label} : +{qty_to_add} {unit} "
                f"({prev_stock} → {new_stock})"
            )
            successes.append(msg)
            log.info(msg)

            if not dry_run:
                state["stock_on_hand"] = new_stock
                if "stock_history" not in state:
                    state["stock_history"] = []
                state["stock_history"].append(history_entry)

        if not dry_run:
            already_processed.add(event.purchase_date)

    return raw_items, successes, warnings


# ── Chargement des credentials depuis l'environnement ─────────────────────────

def _load_gdrive_config() -> tuple[str, str]:
    """Retourne (credentials_path, file_id) depuis les variables d'environnement."""
    creds_path = os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE", "")
    file_id = os.environ.get("GDRIVE_PURCHASES_FILE_ID", "")

    missing = []
    if not creds_path:
        missing.append("GDRIVE_SERVICE_ACCOUNT_FILE")
    if not file_id:
        missing.append("GDRIVE_PURCHASES_FILE_ID")
    if missing:
        raise EnvironmentError(
            f"Variables d'environnement manquantes : {', '.join(missing)}. "
            "Configurez-les dans .env ou secrets.toml."
        )
    return creds_path, file_id


# ── Point d'entrée principal ───────────────────────────────────────────────────

def main():
    """Intègre les achats du fichier Google Drive (ou local) dans stock_items.json."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Intègre les achats du fichier Google Drive dans stock_items.json."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les mises à jour prévues sans modifier stock_items.json.",
    )
    parser.add_argument(
        "--local",
        metavar="FICHIER.xlsx",
        help="Utilise un fichier Excel local au lieu de Google Drive (tests).",
    )
    parser.add_argument(
        "--items",
        metavar="FICHIER.json",
        default=str(STOCK_ITEMS_PATH),
        help="Chemin vers stock_items.json (défaut : stocks/stock_items.json).",
    )
    parser.add_argument(
        "--mapping",
        metavar="FICHIER.json",
        default=str(PURCHASE_MAPPING_PATH),
        help="Chemin vers purchase_mapping.json.",
    )
    args = parser.parse_args()

    items_path = Path(args.items)
    mapping_path = Path(args.mapping)

    load_project_env(required_vars=[])

    # ── 1. Téléchargement ou lecture locale du fichier Excel ───────────────────
    if args.local:
        local_path = Path(args.local)
        if not local_path.exists():
            log.error("Fichier local introuvable : %s", local_path)
            sys.exit(1)
        log.info("Mode local : lecture de %s", local_path)
        excel_bytes = local_path.read_bytes()
    else:
        try:
            creds_path, file_id = _load_gdrive_config()
        except EnvironmentError as exc:
            log.error("%s", exc)
            sys.exit(1)

        try:
            log.info("Téléchargement du fichier Drive '%s'…", file_id)
            excel_bytes = download_file_as_bytes(file_id, creds_path)
        except (FileNotFoundError, PermissionError, RuntimeError) as exc:
            log.error("Erreur Google Drive : %s", exc)
            sys.exit(1)

    # ── 2. Parsing du fichier Excel ────────────────────────────────────────────
    try:
        events = parse_purchases_excel(excel_bytes)
    except (ValueError, ImportError) as exc:
        log.error("Erreur de parsing Excel : %s", exc)
        sys.exit(1)

    if not events:
        log.info("Aucun achat à intégrer. Fin.")
        return

    # ── 3. Chargement des données de stock ────────────────────────────────────
    try:
        raw_items = _load_stock_items_raw(items_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    mapping = load_purchase_mapping(mapping_path)
    already_processed = find_already_processed_dates(raw_items)

    if already_processed:
        log.info(
            "%d date(s) déjà traitée(s) : %s",
            len(already_processed),
            ", ".join(sorted(d.isoformat() for d in already_processed)),
        )

    # ── 4. Application des achats ──────────────────────────────────────────────
    raw_items, successes, warnings = apply_purchases_to_stock(
        raw_items, events, mapping, already_processed, dry_run=args.dry_run
    )

    # ── 5. Rapport de résultat ────────────────────────────────────────────────
    if args.dry_run:
        print("\n=== SIMULATION (--dry-run) — aucune modification effectuée ===\n")
    else:
        print(f"\n=== Achats intégrés : {len(successes)} mise(s) à jour ===\n")

    for line in successes:
        print(f"  + {line}")

    if warnings:
        print(f"\n=== {len(warnings)} avertissement(s) ===\n")
        for w in warnings:
            print(f"  ! {w}")

    # ── 6. Sauvegarde ─────────────────────────────────────────────────────────
    if not args.dry_run and successes:
        _save_stock_items(items_path, raw_items)
        log.info("stock_items.json mis à jour (%d modifications).", len(successes))
        print(f"\nstock_items.json sauvegardé ({len(successes)} modification(s)).")
    elif not args.dry_run:
        log.info("Aucune modification — stock_items.json inchangé.")
        print("\nAucune modification à apporter.")


# app.py l'importe via subprocess; ce bloc sert aussi à l'appel direct
if __name__ == "__main__":
    main()
