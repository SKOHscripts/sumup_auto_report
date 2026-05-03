#!/usr/bin/env python3
"""Registre des modèles entraînés : versions archivées + journal de métriques.

Layout sur disque ::

    stocks/models/
    ├── current.joblib              # symlink vers la dernière version validée
    ├── current.joblib.meta.json    # symlink vers ses métadonnées
    ├── archive/
    │   ├── 2026-W18/
    │   │   ├── model.joblib
    │   │   └── model.joblib.meta.json
    │   └── 2026-W19/...
    └── history.csv                  # journal de toutes les évaluations

Chaque appel à ``promote_model`` écrit une ligne dans ``history.csv`` qu'on
peut consulter pour suivre la dérive de qualité dans le temps.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stocks.ml.evaluation import EvaluationMetrics
from stocks.ml.model import QuantileGradientBoostingForecaster
from utils.sumup_shared import iso_week_label

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARCHIVE_DIR = MODELS_DIR / "archive"
HISTORY_CSV = MODELS_DIR / "history.csv"
CURRENT_MODEL = MODELS_DIR / "current.joblib"

HISTORY_HEADER = [
    "promoted_at",
    "week_label",
    "version",
    "promoted",
    "mae",
    "mape",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "coverage_p10_p90",
    "baseline_mape",
    "n_samples",
    "n_folds",
    "reasons",
]


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _archive_path(week_label: str) -> Path:
    safe = week_label.replace("-", "_")
    return ARCHIVE_DIR / safe / "model.joblib"


def archive_model(model: QuantileGradientBoostingForecaster, week_label: str | None = None) -> Path:
    """Sauve le modèle dans archive/<semaine>/. Retourne le chemin du fichier joblib."""
    _ensure_dirs()
    label = week_label or iso_week_label(datetime.now(timezone.utc))
    target = _archive_path(label)
    target.parent.mkdir(parents=True, exist_ok=True)
    return model.save(target)


def set_current(model_path: Path) -> Path:
    """Pointe ``current.joblib`` vers ``model_path``. Retombe sur copie si symlink impossible."""
    _ensure_dirs()
    meta_src = model_path.with_suffix(model_path.suffix + ".meta.json")
    meta_dst = CURRENT_MODEL.with_suffix(CURRENT_MODEL.suffix + ".meta.json")
    for dst in (CURRENT_MODEL, meta_dst):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    try:
        CURRENT_MODEL.symlink_to(model_path)
        if meta_src.exists():
            meta_dst.symlink_to(meta_src)
    except (OSError, NotImplementedError):
        # Systèmes sans symlink (Windows non admin, certains FS) : on copie.
        import shutil  # pylint: disable=import-outside-toplevel

        shutil.copy2(model_path, CURRENT_MODEL)
        if meta_src.exists():
            shutil.copy2(meta_src, meta_dst)
    return CURRENT_MODEL


def load_current() -> Optional[QuantileGradientBoostingForecaster]:
    """Charge le modèle pointé par ``current.joblib`` ou retourne None si absent."""
    if not CURRENT_MODEL.exists():
        return None
    try:
        return QuantileGradientBoostingForecaster.load(CURRENT_MODEL)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("Impossible de charger le modele courant : %s", exc)
        return None


def append_history(
    metrics: EvaluationMetrics,
    promoted: bool,
    version: str,
    week_label: str,
    baseline_mape: float | None = None,
    reasons: list[str] | None = None,
) -> Path:
    """Ajoute une ligne au journal de promotion (CSV)."""
    _ensure_dirs()
    is_new = not HISTORY_CSV.exists()
    row = {
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "week_label": week_label,
        "version": version,
        "promoted": int(promoted),
        "mae": f"{metrics.mae:.4f}",
        "mape": f"{metrics.mape:.4f}",
        "pinball_q10": f"{metrics.pinball_q10:.4f}",
        "pinball_q50": f"{metrics.pinball_q50:.4f}",
        "pinball_q90": f"{metrics.pinball_q90:.4f}",
        "coverage_p10_p90": f"{metrics.coverage_p10_p90:.4f}",
        "baseline_mape": "" if baseline_mape is None else f"{baseline_mape:.4f}",
        "n_samples": metrics.n_samples,
        "n_folds": metrics.n_folds,
        "reasons": " | ".join(reasons or []),
    }
    with open(HISTORY_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return HISTORY_CSV


def promote_if_better(
    model: QuantileGradientBoostingForecaster,
    metrics: EvaluationMetrics,
    promotable: bool,
    reasons: list[str],
    baseline_mape: float | None = None,
    week_label: str | None = None,
) -> tuple[bool, Path | None]:
    """Archive le modèle, mets à jour ``current`` s'il est promotable, journalise.

    Retourne ``(promu, chemin_archive)``.
    """
    label = week_label or iso_week_label(datetime.now(timezone.utc))
    archive_path = archive_model(model, week_label=label)
    version = model.metadata.config_hash or label
    if promotable:
        set_current(archive_path)
        log.info(
            "ML : modele %s promu (MAPE=%.2f%%, coverage=%.0f%%)",
            version, metrics.mape * 100, metrics.coverage_p10_p90 * 100,
        )
    else:
        log.warning(
            "ML : modele %s NON promu : %s (MAPE=%.2f%%, coverage=%.0f%%)",
            version, " | ".join(reasons), metrics.mape * 100, metrics.coverage_p10_p90 * 100,
        )
    append_history(
        metrics=metrics,
        promoted=promotable,
        version=version,
        week_label=label,
        baseline_mape=baseline_mape,
        reasons=reasons,
    )
    return promotable, archive_path


def recent_history(n: int = 10) -> list[dict]:
    """Retourne les ``n`` dernières lignes du journal (utile pour drift detection)."""
    if not HISTORY_CSV.exists():
        return []
    with open(HISTORY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)[-n:]


def detect_drift(
    n: int = 3,
    mape_threshold: float = 0.45,
) -> tuple[bool, str]:
    """Vrai si les ``n`` derniers modèles ont une MAPE > seuil consécutivement."""
    rows = recent_history(n=n)
    if len(rows) < n:
        return False, f"journal trop court ({len(rows)} < {n})"
    breached = []
    for row in rows:
        try:
            current_mape = float(row["mape"])
        except (KeyError, ValueError):
            continue
        if current_mape > mape_threshold:
            breached.append(f"{row.get('week_label', '?')}={current_mape:.2%}")
    if len(breached) == n:
        return True, f"MAPE > {mape_threshold:.0%} sur {n} semaines ({', '.join(breached)})"
    return False, "qualite stable"


__all__ = [
    "ARCHIVE_DIR",
    "CURRENT_MODEL",
    "HISTORY_CSV",
    "MODELS_DIR",
    "append_history",
    "archive_model",
    "detect_drift",
    "load_current",
    "promote_if_better",
    "recent_history",
    "set_current",
]
