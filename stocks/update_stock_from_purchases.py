#!/usr/bin/env python3
"""
Mise à jour des stocks depuis le fichier d'achats Google Drive.
──────────────────────────────────────────────────────────────
Télécharge le fichier Excel ACHATS_suivi_stock.xlsx depuis Google Drive,
parse les colonnes d'achat (date + acheteur + quantités), et met à jour
stock_items.json en ajoutant les quantités achetées au stock_on_hand.

Deux types de colonnes sont reconnus, distingués par la ligne 2 (la
même que les marqueurs « exemple ») :

  - Colonne d'achat (par défaut) : les quantités sont AJOUTÉES au stock.
  - Colonne « état des lieux » : si la ligne 2 contient « état des lieux »
    (ou « inventaire »), les quantités REMPLACENT le stock enregistré.
    Cela permet à n'importe qui de déclarer, directement depuis l'Excel,
    qu'un comptage physique a été réalisé à cette date. La date de
    l'état des lieux ré-ancre aussi le calcul de consommation du rapport.

Usage :
  python -m stocks.update_stock_from_purchases            # depuis Google Drive
  python -m stocks.update_stock_from_purchases --dry-run  # simulation sans écriture
  python -m stocks.update_stock_from_purchases --local FICHIER.xlsx

Variables d'environnement requises (sauf --local) :
  GDRIVE_SERVICE_ACCOUNT_JSON  Contenu JSON du service account (prioritaire, pour Streamlit)
  GDRIVE_SERVICE_ACCOUNT_FILE  Chemin vers le JSON du service account (CLI / crontab)
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

from stocks.gdrive_loader import download_file_as_bytes, download_file_as_bytes_from_info
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

# Marqueurs (ligne 2 de l'Excel, déjà normalisés sans accents) qui déclarent
# une colonne comme un « état des lieux » : les quantités remplacent le stock
# au lieu d'y être ajoutées.
INVENTORY_MARKERS = ("etat des lieux", "etat des stocks", "inventaire", "inventory", "snapshot")

# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class PurchaseItem:
    """Un produit compté dans une colonne du fichier Excel."""

    excel_label: str
    qty: float


@dataclass
class PurchaseEvent:
    """Une colonne du fichier Excel.

    kind vaut "purchase" (les quantités sont ajoutées au stock) ou
    "inventory" (état des lieux : les quantités remplacent le stock).
    """

    purchase_date: date
    buyer: str
    items: list[PurchaseItem] = field(default_factory=list)
    kind: str = "purchase"


def _is_inventory_marker(marker_norm: str) -> bool:
    """Indique si un marqueur (déjà normalisé) déclare un état des lieux."""
    return any(m in marker_norm for m in INVENTORY_MARKERS)


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
      Ligne 2 : marqueurs de colonne :
        - "exemple"         → colonne ignorée
        - "état des lieux"  → colonne d'inventaire (remplace le stock)
        - (vide)            → colonne d'achat (ajoute au stock)
      Ligne 3 : prénoms acheteurs  (col C+)
      Ligne 4 : dates d'achat      (col C+)
      Ligne 5 : vide
      Ligne 6+ : données produits par catégorie
        col A : type d'unité (propagé vers le bas)
        col B : nom catégorie (MAJUSCULES) ou nom produit
        col C+: quantités achetées ou comptées
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

    # Identifie les colonnes de données (à partir de la colonne C = index 2),
    # filtre les colonnes "exemple" et distingue achats / états des lieux.
    purchase_cols: list[tuple[int, date, str, str]] = []  # (col_idx, date, buyer, kind)
    for col_idx in range(2, len(row_dates)):
        marker_norm = ""
        if col_idx < len(row_example):
            marker_norm = normalize(str(row_example[col_idx] or ""))
            if "exemple" in marker_norm:
                continue

        raw_date = row_dates[col_idx] if col_idx < len(row_dates) else None
        if raw_date is None:
            continue

        purchase_date = _parse_excel_date(raw_date)
        if purchase_date is None:
            continue

        buyer = str("").strip() if col_idx < len(row_buyers) else ""  # anonymisé volontairement
        kind = "inventory" if _is_inventory_marker(marker_norm) else "purchase"
        purchase_cols.append((col_idx, purchase_date, buyer, kind))

    if not purchase_cols:
        log.warning("Aucune colonne d'achat valide trouvée dans le fichier Excel.")
        return []

    events: dict[int, PurchaseEvent] = {
        col_idx: PurchaseEvent(purchase_date=d, buyer=b, kind=kind)
        for col_idx, d, b, kind in purchase_cols
    }

    for row in rows[6:]:
        col_b = str(row[1] or "").strip() if len(row) > 1 else ""

        if not col_b:
            continue

        # Ignore les en-têtes de catégorie (tout en majuscules)
        if col_b == col_b.upper() and col_b.replace("/", "").replace("&", "").replace(" ", "").isupper():
            continue

        for col_idx, _, _, kind in purchase_cols:
            raw_qty = row[col_idx] if col_idx < len(row) else None
            if raw_qty is None:
                continue
            try:
                qty = float(raw_qty)
            except (TypeError, ValueError):
                continue
            if qty < 0:
                continue
            # Pour un achat, une quantité nulle = rien acheté → ignorée.
            # Pour un état des lieux, 0 est un comptage valide (stock vide).
            if qty == 0 and kind != "inventory":
                continue
            events[col_idx].items.append(PurchaseItem(excel_label=col_b, qty=qty))

    result = [ev for ev in events.values() if ev.items]
    n_inv = sum(1 for ev in result if ev.kind == "inventory")
    log.info(
        "%d colonne(s) détectée(s) dans le fichier Excel (%d achat(s), %d état(s) des lieux).",
        len(result), len(result) - n_inv, n_inv,
    )
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
    """Retourne les dates déjà intégrées via stock_history.

    Couvre aussi bien les achats (type='purchase') que les états des lieux
    (type='inventory'), afin qu'une colonne déjà traitée ne soit pas
    réappliquée à chaque exécution.
    """
    processed: set[date] = set()
    for item in raw_items:
        state = item.get("stock_state") or {}
        for entry in state.get("stock_history", []):
            if entry.get("type") in ("purchase", "inventory"):
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
    """Applique les événements (achats et états des lieux) sur raw_items.

    Pour chaque événement dont la date n'est pas déjà traitée :
      - Résout chaque excel_label en stock_sku via le mapping
      - Achat (kind='purchase') : ajoute qty * multiplicateur au
        stock_on_hand et trace une entrée stock_history 'purchase'.
      - État des lieux (kind='inventory') : remplace stock_on_hand par
        qty * multiplicateur, ré-ancre last_inventory_date / last_auto_update
        à la date du comptage et trace une entrée stock_history 'inventory'.

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
        is_inventory = event.kind == "inventory"
        log.info(
            "Traitement %s %s par %s (%d produit(s)).",
            "état des lieux" if is_inventory else "achat",
            date_str, event.buyer or "?", len(event.items),
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

            qty_value = purchase_item.qty * multiplier
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
            label = ref_item.get("stock_label") or ref_item.get("label") or stock_sku
            unit = ref_item.get("stock_unit") or ref_item.get("unit") or ""

            if is_inventory:
                new_stock = round(qty_value, 6)
                history_entry = {
                    "type": "inventory",
                    "date": date_str,
                    "buyer": event.buyer or "",
                    "counted_qty": qty_value,
                    "previous_stock_on_hand": prev_stock,
                    "new_stock_on_hand": new_stock,
                    "source": "gdrive_excel",
                }
                msg = (
                    f"[{date_str}] {label} : état des lieux = {new_stock} {unit} "
                    f"({prev_stock} → {new_stock})"
                )
            else:
                new_stock = round(prev_stock + qty_value, 6)
                history_entry = {
                    "type": "purchase",
                    "date": date_str,
                    "buyer": event.buyer or "",
                    "qty_added": qty_value,
                    "previous_stock_on_hand": prev_stock,
                    "new_stock_on_hand": new_stock,
                    "source": "gdrive_excel",
                }
                msg = (
                    f"[{date_str}] {label} : +{qty_value} {unit} "
                    f"({prev_stock} → {new_stock})"
                )

            successes.append(msg)
            log.info(msg)

            if not dry_run:
                state["stock_on_hand"] = new_stock
                if is_inventory:
                    # Le comptage physique fait foi à cette date : on ré-ancre
                    # le calcul de consommation du rapport (auto_refresh) afin
                    # que seules les ventes postérieures soient déduites.
                    state["last_inventory_date"] = date_str
                    state["inventory_count_method"] = "manual"
                    state["last_auto_update"] = date_str
                if "stock_history" not in state:
                    state["stock_history"] = []
                state["stock_history"].append(history_entry)

        if not dry_run:
            already_processed.add(event.purchase_date)

    return raw_items, successes, warnings


# ── Chargement des credentials depuis l'environnement ─────────────────────────

def _load_gdrive_config() -> tuple[str | dict, str]:
    """Retourne (credentials, file_id) depuis les variables d'environnement.

    credentials est soit un dict (si GDRIVE_SERVICE_ACCOUNT_JSON est défini)
    soit un chemin de fichier str (si GDRIVE_SERVICE_ACCOUNT_FILE est défini).
    GDRIVE_SERVICE_ACCOUNT_JSON est prioritaire (utilisé par Streamlit).
    """
    file_id = os.environ.get("GDRIVE_PURCHASES_FILE_ID", "")
    if not file_id:
        raise EnvironmentError(
            "Variable d'environnement manquante : GDRIVE_PURCHASES_FILE_ID. "
            "Configurez-la dans .env ou secrets.toml."
        )

    sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "")
    if sa_json:
        try:
            return json.loads(sa_json), file_id
        except json.JSONDecodeError as exc:
            raise EnvironmentError(
                "GDRIVE_SERVICE_ACCOUNT_JSON n'est pas un JSON valide."
            ) from exc

    creds_path = os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE", "")
    if not creds_path:
        raise EnvironmentError(
            "Aucune credentials Google Drive configurée. "
            "Définissez GDRIVE_SERVICE_ACCOUNT_JSON (Streamlit) "
            "ou GDRIVE_SERVICE_ACCOUNT_FILE (CLI/cron) dans .env ou secrets.toml."
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
            creds, file_id = _load_gdrive_config()
        except EnvironmentError as exc:
            log.error("%s", exc)
            sys.exit(1)

        try:
            log.info("Téléchargement du fichier Drive '%s'…", file_id)
            if isinstance(creds, dict):
                excel_bytes = download_file_as_bytes_from_info(file_id, creds)
            else:
                excel_bytes = download_file_as_bytes(file_id, creds)
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
        print(f"\n=== Mises à jour intégrées (achats + états des lieux) : {len(successes)} ===\n")

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
