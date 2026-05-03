#!/usr/bin/env python3
"""CLI d'entraînement et de promotion d'un modèle quantile.

Usage :
  python -m stocks.ml.train                 # entraîne, évalue, promeut si OK
  python -m stocks.ml.train --force         # promeut même si seuils non atteints
  python -m stocks.ml.train --no-promote    # entraîne et évalue sans rien archiver
  python -m stocks.ml.train --report        # affiche les 10 dernières évaluations

À lancer typiquement chaque semaine, avant la génération du rapport. Idempotent
et résistant aux historiques courts (sortie propre + journal).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Permet ``python stocks/ml/train.py`` ou ``python -m stocks.ml.train``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stocks.ml.dataset import load_weekly_usage  # noqa: E402
from stocks.ml.evaluation import (  # noqa: E402
    DEFAULT_MAPE_THRESHOLD,
    baseline_avg_rolling4,
    is_model_promotable,
    walk_forward_backtest,
)
from stocks.ml.model import QuantileGradientBoostingForecaster  # noqa: E402
from stocks.ml.registry import (  # noqa: E402
    detect_drift,
    promote_if_better,
    recent_history,
)
from stocks.ml.features import prepare_training_table  # noqa: E402
from utils.sumup_shared import iso_week_label  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stocks.ml.train")


def run(force: bool = False, do_promote: bool = True, max_iter: int = 200) -> int:
    """Pipeline complet : load → backtest → train final → promote → log."""
    history = load_weekly_usage()
    if len(history) == 0:
        log.error("Aucun historique persistant. Lancer 'python -m stocks.ml.bootstrap' avant.")
        return 1

    log.info("Backtest walk-forward (5 plis)...")
    metrics = walk_forward_backtest(history, n_folds=5, max_iter=max_iter)
    if metrics.n_folds == 0:
        log.error("Pas assez de donnees pour evaluer. Continuez d'accumuler l'historique.")
        return 2

    baseline = baseline_avg_rolling4(history)
    promotable, reasons = is_model_promotable(metrics, baseline_mape=baseline)
    import math  # pylint: disable=import-outside-toplevel

    baseline_pct = baseline * 100 if not math.isnan(baseline) else float("nan")
    log.info(
        "Metriques agregees : MAPE=%.2f%% MAE=%.2f coverage=%.0f%% (baseline_mape=%.2f%%)",
        metrics.mape * 100, metrics.mae, metrics.coverage_p10_p90 * 100, baseline_pct,
    )
    if reasons:
        log.warning("Raisons de non-promotion : %s", " | ".join(reasons))

    log.info("Entrainement du modele final sur tout l'historique...")
    X, y, _ = prepare_training_table(history)
    final_model = QuantileGradientBoostingForecaster(max_iter=max_iter).fit(X, y)

    if not do_promote:
        log.info("--no-promote : pas d'ecriture en archive ni de mise a jour de current.")
        return 0

    week = iso_week_label(datetime.now(timezone.utc))
    promoted, _archive = promote_if_better(
        model=final_model,
        metrics=metrics,
        promotable=force or promotable,
        reasons=reasons if not force else ["promotion forcee"],
        baseline_mape=baseline,
        week_label=week,
    )

    drifted, drift_msg = detect_drift(n=3, mape_threshold=DEFAULT_MAPE_THRESHOLD)
    if drifted:
        log.error("ALERTE DRIFT : %s", drift_msg)
        return 3

    return 0 if promoted else 4


def show_report(n: int = 10) -> int:
    """Affiche les ``n`` dernières lignes du journal de promotion."""
    rows = recent_history(n=n)
    if not rows:
        log.info("Aucun historique de promotion (lancer 'train' une fois).")
        return 0
    print("Dernières évaluations :")
    print(f"{'date':<20} {'week':<10} {'promu':<6} {'MAPE':<8} {'coverage':<10} {'baseline':<10}")
    for row in rows:
        promoted = "OUI" if row["promoted"] == "1" else "NON"
        baseline = row["baseline_mape"] or "-"
        print(
            f"{row['promoted_at']:<20} {row['week_label']:<10} {promoted:<6} "
            f"{float(row['mape']):.2%}   {float(row['coverage_p10_p90']):.0%}      {baseline}"
        )
    return 0


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Promeut même si seuils non atteints")
    parser.add_argument("--no-promote", action="store_true", help="N'archive pas et ne met pas à jour current")
    parser.add_argument("--max-iter", type=int, default=200, help="Itérations HGB (défaut : 200)")
    parser.add_argument("--report", action="store_true", help="Affiche les 10 dernières évaluations")
    args = parser.parse_args()

    if args.report:
        sys.exit(show_report())
    sys.exit(run(force=args.force, do_promote=not args.no_promote, max_iter=args.max_iter))


if __name__ == "__main__":
    main()
