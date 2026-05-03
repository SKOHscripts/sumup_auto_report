#!/usr/bin/env python3
"""Diagnostic par SKU pour identifier les SKU qui plombent les métriques globales.

Calcule pour chaque SKU :
  - n_weeks       : nombre de semaines avec donnée
  - n_zeros       : nombre de semaines à zéro consommation
  - pct_zeros     : proportion de semaines à zéro
  - mean_usage    : moyenne hebdo
  - std_usage     : écart-type hebdo (volatilité)
  - cv            : coefficient de variation (std/mean) — au-dessus de 1.5 c'est très volatil
  - last_4w_mean  : moyenne sur les 4 dernières semaines
  - mape_naive    : MAPE de la baseline "prédit = lag-1"
  - mape_avg4     : MAPE de la baseline "prédit = avg 4 sem précédentes"

Utilisable via la CLI ``python -m stocks.ml.train --diagnose``. Le rapport
est imprimé en TSV (paste-friendly) et peut optionnellement être sauvegardé
en CSV.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stocks.ml.evaluation import mape

log = logging.getLogger(__name__)


def _per_sku_metrics(group: pd.DataFrame) -> dict:
    """Calcule les métriques pour un SKU (DataFrame trié chronologiquement)."""
    g = group.sort_values(["year", "week"]).reset_index(drop=True)
    usage = g["usage"].astype("float64").to_numpy()
    n = len(usage)
    n_zeros = int(np.sum(usage == 0))
    mean_u = float(np.mean(usage)) if n else 0.0
    std_u = float(np.std(usage, ddof=1)) if n > 1 else 0.0
    cv = std_u / mean_u if mean_u > 0 else float("inf") if std_u > 0 else 0.0

    last_4 = usage[-4:] if n >= 1 else np.array([])
    last_4w_mean = float(np.mean(last_4)) if len(last_4) else 0.0

    # Baselines pour MAPE individuelle (avec eps=1 pour gérer les zéros).
    mape_naive = float("nan")
    mape_avg4 = float("nan")
    if n >= 2:
        y_true = usage[1:]
        y_pred = usage[:-1]
        mape_naive = mape(y_true, y_pred)
    if n >= 5:
        # Pour t in [4..n-1], pred = mean(usage[t-4..t-1])
        rolled = pd.Series(usage).rolling(window=4, min_periods=1).mean().shift(1).to_numpy()
        valid = ~np.isnan(rolled)
        if valid.sum() >= 1:
            mape_avg4 = mape(usage[valid], rolled[valid])
    return {
        "stock_sku": g["stock_sku"].iloc[0],
        "n_weeks": n,
        "n_zeros": n_zeros,
        "pct_zeros": (n_zeros / n) if n else 0.0,
        "mean_usage": mean_u,
        "std_usage": std_u,
        "cv": cv,
        "last_4w_mean": last_4w_mean,
        "mape_naive": mape_naive,
        "mape_avg4": mape_avg4,
    }


def diagnose(history_df: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un DataFrame avec une ligne par SKU, trié par mape_avg4 décroissante.

    Les SKU les plus difficiles (forte MAPE) apparaissent en haut.
    """
    if len(history_df) == 0:
        return pd.DataFrame()
    rows = [_per_sku_metrics(g) for _, g in history_df.groupby("stock_sku", sort=False)]
    df = pd.DataFrame(rows)
    return df.sort_values(["mape_avg4", "cv"], ascending=[False, False], na_position="last").reset_index(drop=True)


def format_table(df: pd.DataFrame, top_n: int | None = None) -> str:
    """Formate le rapport en table texte alignée."""
    if len(df) == 0:
        return "(aucun SKU dans l'historique)"
    show = df if top_n is None else df.head(top_n)
    headers = [
        "SKU", "n_sem", "n_0", "%_0", "mean", "std", "CV",
        "moy_4sem", "MAPE_naive", "MAPE_avg4",
    ]
    rows = []
    for _, r in show.iterrows():
        rows.append([
            str(r["stock_sku"])[:24],
            f"{int(r['n_weeks'])}",
            f"{int(r['n_zeros'])}",
            f"{r['pct_zeros']:.0%}",
            f"{r['mean_usage']:.2f}",
            f"{r['std_usage']:.2f}",
            f"{r['cv']:.2f}" if np.isfinite(r["cv"]) else "inf",
            f"{r['last_4w_mean']:.2f}",
            "nan" if np.isnan(r["mape_naive"]) else f"{r['mape_naive']:.0%}",
            "nan" if np.isnan(r["mape_avg4"]) else f"{r['mape_avg4']:.0%}",
        ])
    widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "  "

    def fmt_row(values):
        return sep.join(v.ljust(w) for v, w in zip(values, widths))

    lines = [fmt_row(headers), sep.join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Sauve le diagnostic en CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


__all__ = ["diagnose", "format_table", "save_csv"]
