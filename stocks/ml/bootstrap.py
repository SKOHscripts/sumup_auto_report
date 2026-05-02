#!/usr/bin/env python3
"""Bootstrap initial du parquet d'historique hebdomadaire.

Récupère depuis l'API SumUp toutes les transactions sur une fenêtre étendue
(par défaut depuis le 2025-12-01) puis alimente
``stocks/data/weekly_usage.parquet``. À lancer une seule fois pour amorcer
l'historique ML, ensuite le pipeline ``sumup_stocks.py`` s'occupe des updates
hebdomadaires incrémentaux.

Usage :
  python -m stocks.ml.bootstrap                       # depuis 2025-12-01
  python -m stocks.ml.bootstrap --since 2025-09-01    # date personnalisée
  python -m stocks.ml.bootstrap --mock mock.json      # depuis un dump local
  python -m stocks.ml.bootstrap --dry-run             # n'écrit pas le parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Permet ``python stocks/ml/bootstrap.py`` ou ``python -m stocks.ml.bootstrap``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stocks.sumup_stocks import (  # noqa: E402  pylint: disable=wrong-import-position
    aggregate_weekly_sales,
    aggregate_weekly_stock_usage,
    build_sku_index,
    enrich_transactions,
    fetch_transactions,
    load_stock_items_raw,
    prepare_enabled_stock_items,
)
from stocks.ml.dataset import DATASET_PATH, update_weekly_usage  # noqa: E402
from utils.mail_utils import load_project_env  # noqa: E402
from utils.sumup_shared import iso_week_label  # noqa: E402

DEFAULT_SINCE = "2025-12-01"
BASE_DIR = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stocks.ml.bootstrap")


def _build_weeks_range(start: datetime, end: datetime) -> list[str]:
    weeks: list[str] = []
    seen = set()
    cursor = start
    while cursor <= end:
        lbl = iso_week_label(cursor)
        if lbl not in seen:
            weeks.append(lbl)
            seen.add(lbl)
        cursor += timedelta(days=7)
    return sorted(seen)


def run(
    since: str = DEFAULT_SINCE,
    items_file: Path | None = None,
    mock_file: str | None = None,
    enrich: bool = True,
    dry_run: bool = False,
    output: Path | None = None,
) -> Path | None:
    """Exécute le bootstrap. Retourne le chemin du parquet écrit, ou None en dry-run."""
    items_file = items_file or BASE_DIR / "stocks" / "stock_items.json"

    start_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc)

    log.info("Bootstrap historique : %s -> %s", start_dt.date(), end_dt.date())

    raw_items = load_stock_items_raw(items_file)
    stock_items = prepare_enabled_stock_items(raw_items)
    sku_index = build_sku_index(stock_items)

    weeks_range = _build_weeks_range(start_dt, end_dt)
    log.info("Fenêtre : %d semaines ISO", len(weeks_range))

    fetch_start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    txns = fetch_transactions(fetch_start, fetch_end, mock_file=mock_file)
    log.info("Transactions brutes récupérées : %d", len(txns))

    if enrich and not mock_file:
        import os  # pylint: disable=import-outside-toplevel

        load_project_env(required_vars=["SUMUP_API_KEY"], logger=log)
        headers = {"Authorization": f"Bearer {os.environ['SUMUP_API_KEY']}"}
        txns = enrich_transactions(txns, headers)

    weekly_sales, unmapped = aggregate_weekly_sales(txns, sku_index, weeks_range)
    weekly_usage, weekly_sales_count = aggregate_weekly_stock_usage(
        stock_items, weekly_sales, weeks_range,
    )
    if unmapped:
        log.warning("%d produit(s) SumUp non mappé(s)", len(unmapped))

    if dry_run:
        log.info("[DRY-RUN] %d SKU x %d semaines à persister", len(weekly_usage), len(weeks_range))
        for sku, by_week in list(weekly_usage.items())[:5]:
            log.info("  %s : %s", sku, dict(list(by_week.items())[:5]))
        return None

    target = output or DATASET_PATH
    merged = update_weekly_usage(weekly_usage, weekly_sales_count, path=target)
    log.info(
        "Parquet écrit : %s (%d lignes, %d SKU)",
        target, len(merged), merged["stock_sku"].nunique(),
    )
    return target


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default=DEFAULT_SINCE, help=f"Date ISO de début (défaut : {DEFAULT_SINCE})")
    parser.add_argument("--items", type=Path, default=None, help="Chemin vers stock_items.json")
    parser.add_argument("--mock", default=None, help="Fichier JSON de transactions (mode hors ligne)")
    parser.add_argument("--no-enrich", action="store_true", help="Ne pas appeler l'API d'enrichissement")
    parser.add_argument("--dry-run", action="store_true", help="N'écrit pas le parquet")
    parser.add_argument("--output", type=Path, default=None, help="Chemin de sortie du parquet")
    args = parser.parse_args()

    run(
        since=args.since,
        items_file=args.items,
        mock_file=args.mock,
        enrich=not args.no_enrich,
        dry_run=args.dry_run,
        output=args.output,
    )


if __name__ == "__main__":
    main()
