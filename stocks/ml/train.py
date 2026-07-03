#!/usr/bin/env python3
"""CLI d'entraînement, tuning, diagnostic et promotion du modèle quantile.

Sous-commandes (toutes équivalentes à un flag pour la compatibilité avec l'ancienne forme) :

  python -m stocks.ml.train                    # train + évaluation + promotion
  python -m stocks.ml.train --tune             # tuning RandomizedSearch puis train
  python -m stocks.ml.train --diagnose         # rapport par SKU (n_sem, %_0, MAPE_avg4)
  python -m stocks.ml.train --report           # 10 dernières évaluations (journal)
  python -m stocks.ml.train --force            # promeut même si seuils non atteints
  python -m stocks.ml.train --no-promote       # train + évaluation, pas d'archive

Flags configurables (s'écrivent aussi dans config.json pour les runs suivants) :

  --quantiles 0.05,0.5,0.95            # défaut : 0.05/0.5/0.95
  --mape-threshold 0.45                # défaut : 0.45
  --coverage-target 0.80               # défaut : 0.80
  --coverage-tolerance 0.10            # défaut : 0.10

Idempotent et résistant aux historiques courts (sortie propre + journal).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Permet ``python stocks/ml/train.py`` ou ``python -m stocks.ml.train``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stocks.ml.config import MLConfig, load_config, save_config  # noqa: E402
from stocks.ml.dataset import load_weekly_usage  # noqa: E402
from stocks.ml.diagnose import diagnose, format_table, save_csv  # noqa: E402
from stocks.ml.evaluation import (  # noqa: E402
    baseline_avg_rolling4,
    is_model_promotable,
    walk_forward_backtest,
)
from stocks.ml.features import prepare_training_table  # noqa: E402
from stocks.ml.model import QuantileGradientBoostingForecaster  # noqa: E402
from stocks.ml.registry import (  # noqa: E402
    detect_drift,
    promote_if_better,
    recent_history,
)
from stocks.ml.tuning import tune_and_save  # noqa: E402
from utils.sumup_shared import iso_week_label  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stocks.ml.train")


def _build_model(cfg: MLConfig, random_state: int = 0) -> QuantileGradientBoostingForecaster:
    """Instancie un modèle quantile à partir de la config (paramètres tunés inclus)."""

    return QuantileGradientBoostingForecaster(
        quantiles=cfg.quantiles,
        random_state=random_state,
        target_transform=cfg.target_transform,
        **cfg.tuned_params,
    )


def run_train(
    cfg: MLConfig,
    force: bool = False,
    do_promote: bool = True,
) -> int:
    """Pipeline complet : load → backtest → train final → promote → log."""
    history = load_weekly_usage()

    if len(history) == 0:
        log.error("Aucun historique persistant. Lancer 'python -m stocks.ml.bootstrap' avant.")

        return 1

    log.info("Backtest walk-forward (5 plis) avec quantiles=%s...", cfg.quantiles)
    metrics = walk_forward_backtest(
        history,
        n_folds=5,
        quantiles=cfg.quantiles,
        target_transform=cfg.target_transform,
        model_params=cfg.tuned_params,
    )

    if metrics.n_folds == 0:
        log.error("Pas assez de donnees pour evaluer. Continuez d'accumuler l'historique.")

        return 2

    baseline = baseline_avg_rolling4(history)
    promotable, reasons = is_model_promotable(
        metrics,
        baseline_mape=baseline,
        mape_threshold=cfg.mape_threshold,
        coverage_target=cfg.coverage_target,
        coverage_tolerance=cfg.coverage_tolerance,
        relative_mape_margin=cfg.relative_mape_margin,
    )
    baseline_pct = baseline * 100 if not math.isnan(baseline) else float("nan")
    log.info(
        "Metriques agregees : MAPE=%.2f%% MAE=%.2f coverage=%.0f%% (baseline_mape=%.2f%%)",
        metrics.mape * 100, metrics.mae, metrics.coverage_band * 100, baseline_pct,
    )

    if reasons:
        log.warning("Raisons de non-promotion : %s", " | ".join(reasons))

    log.info("Entrainement du modele final sur tout l'historique...")
    X, y, _ = prepare_training_table(history)
    final_model = _build_model(cfg)
    final_model.fit(X, y)

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

    drifted, drift_msg = detect_drift(n=3, mape_threshold=cfg.mape_threshold)

    if drifted:
        log.warning("ALERTE DRIFT : %s", drift_msg)

        return 3

    return 0 if promoted else 4


def run_tune(n_candidates: int = 300, n_jobs: int = -1, exhaustive: bool = False) -> int:
    """Tuning par successive halving puis sauvegarde de la config + train final."""
    history = load_weekly_usage()

    if len(history) == 0:
        log.error("Aucun historique persistant. Lancer 'python -m stocks.ml.bootstrap' avant.")

        return 1
    log.info(
        "Tuning des hyperparametres (%s, n_jobs=%s)...",
        "grille exhaustive" if exhaustive else f"n_candidates={n_candidates}", n_jobs,
    )
    new_cfg = tune_and_save(history, n_candidates=n_candidates, n_jobs=n_jobs, exhaustive=exhaustive)
    log.info(
        "Config : %s (params=%s, score_pinball=%.4f)",
        "stocks/models/config.json", new_cfg.tuned_params, new_cfg.tuning_score or 0.0,
    )
    # On lance ensuite un train normal pour valider la config tunee.

    return run_train(new_cfg, force=False, do_promote=True)


def run_diagnose(output_csv: Path | None = None) -> int:
    """Affiche un rapport par SKU (et écrit éventuellement en CSV)."""
    history = load_weekly_usage()

    if len(history) == 0:
        log.error("Aucun historique persistant. Lancer 'python -m stocks.ml.bootstrap' avant.")

        return 1
    df = diagnose(history)
    print(format_table(df))

    if output_csv:
        path = save_csv(df, output_csv)
        log.info("Rapport CSV ecrit : %s", path)

    return 0


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
        try:
            baseline = f"{float(row['baseline_mape']):.2%}"
        except (TypeError, ValueError):
            baseline = "-"
        print(
            f"{row['promoted_at']:<20} {row['week_label']:<10} {promoted:<6} "
            f"{float(row['mape']):.2%}   {float(row['coverage_band']):.0%}      {baseline}"
        )

    return 0


def _parse_quantiles(raw: str) -> tuple[float, float, float]:
    """Parse une chaîne '0.05,0.5,0.95' en tuple ordonné."""
    parts = [float(p.strip()) for p in raw.split(",")]

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--quantiles doit avoir 3 valeurs, recu : {raw}")
    parts.sort()

    if abs(parts[1] - 0.5) > 1e-6:
        raise argparse.ArgumentTypeError("Le quantile median doit valoir 0.5")

    return (parts[0], parts[1], parts[2])


def _apply_overrides(cfg: MLConfig, args) -> MLConfig:
    """Applique les flags de seuil/quantiles sur la config et la persiste si modifiée."""
    changed = False

    if args.quantiles is not None:
        cfg.quantiles = args.quantiles
        changed = True

    if args.mape_threshold is not None:
        cfg.mape_threshold = args.mape_threshold
        changed = True

    if args.coverage_target is not None:
        cfg.coverage_target = args.coverage_target
        changed = True

    if args.coverage_tolerance is not None:
        cfg.coverage_tolerance = args.coverage_tolerance
        changed = True

    if changed:
        save_config(cfg)
        log.info("Config mise a jour et persistee : %s", cfg.as_dict())

    return cfg


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Sous-commandes (mutuellement exclusives en pratique).
    parser.add_argument("--tune", action="store_true", help="Lance la recherche d'hyperparametres avant de train")
    parser.add_argument("--diagnose", action="store_true", help="Affiche un rapport par SKU et sort")
    parser.add_argument("--report", action="store_true", help="Affiche les 10 dernieres evaluations et sort")

    # Modificateurs du train.
    parser.add_argument("--force", action="store_true", help="Promeut meme si seuils non atteints")
    parser.add_argument("--no-promote", action="store_true", help="N'archive pas et ne met pas a jour current")

    # Tuning.
    parser.add_argument("--n-candidates", type=int, default=None,
                        help="Nombre de combinaisons echantillonnees par le halving (defaut : 300)")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Nombre de coeurs pour le tuning (-1 = tous ; defaut : -1)")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Tuning : balaye TOUTE la grille (HalvingGridSearchCV) au lieu d'un echantillon")

    # Configurables (persistes dans config.json).
    parser.add_argument("--quantiles", type=_parse_quantiles, default=None,
                        help="Triplet 'q_low,q_med,q_high' ex: '0.05,0.5,0.95'")
    parser.add_argument("--mape-threshold", type=float, default=None, help="Seuil MAPE max (defaut config)")
    parser.add_argument("--coverage-target", type=float, default=None, help="Cible de couverture (defaut config)")
    parser.add_argument("--coverage-tolerance", type=float, default=None, help="Tolerance autour de la cible")

    # Diagnose.
    parser.add_argument("--diagnose-csv", type=Path, default=None,
                        help="Avec --diagnose : sauve aussi le rapport en CSV")

    args = parser.parse_args()

    if args.report:
        sys.exit(show_report())

    cfg = load_config()
    cfg = _apply_overrides(cfg, args)

    if args.diagnose:
        sys.exit(run_diagnose(args.diagnose_csv))

    if args.tune:
        run_tune(n_candidates=args.n_candidates or 300, n_jobs=args.jobs,
                 exhaustive=args.exhaustive)
    run_train(cfg, force=args.force, do_promote=not args.no_promote)


if __name__ == "__main__":
    main()
