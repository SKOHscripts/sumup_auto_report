#!/usr/bin/env python3
"""Détection d'anomalies dans l'historique de consommation hebdomadaire.

Une semaine est marquée anomalie si son z-score (par SKU) dépasse ``k`` :
    z = |usage - mean| / std  >  k

Les anomalies peuvent être :
  - affichées sur le graphe historique (marqueur rouge) ;
  - exclues de l'entraînement si ``exclude=True``.

Usage typique :
    from stocks.ml.anomaly import detect_anomalies
    df_flagged = detect_anomalies(history_df, k=2.5)
    clean_df = df_flagged[~df_flagged["is_anomaly"]]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_Z_THRESHOLD = 2.5
# Nombre minimum de semaines par SKU pour calculer un z-score fiable.
MIN_WEEKS_FOR_ZSCORE = 4


def detect_anomalies(
    history_df: pd.DataFrame,
    k: float = DEFAULT_Z_THRESHOLD,
) -> pd.DataFrame:
    """Ajoute les colonnes ``z_score`` et ``is_anomaly`` à l'historique.

    Args:
        history_df: DataFrame avec au moins ``stock_sku``, ``usage``.
        k: seuil de z-score au-delà duquel une semaine est anomalie.

    Returns:
        Copie du DataFrame avec deux nouvelles colonnes :
          - ``z_score`` (float, NaN si pas assez de données) ;
          - ``is_anomaly`` (bool).
    """
    df = history_df.copy()
    df["z_score"] = float("nan")
    df["is_anomaly"] = False

    for sku, grp in df.groupby("stock_sku"):
        if len(grp) < MIN_WEEKS_FOR_ZSCORE:
            continue
        usage = grp["usage"].to_numpy(dtype=float)
        mu = float(np.mean(usage))
        sigma = float(np.std(usage, ddof=1))
        if sigma == 0.0:
            continue
        z = np.abs(usage - mu) / sigma
        df.loc[grp.index, "z_score"] = z
        df.loc[grp.index, "is_anomaly"] = z > k

    return df


def anomaly_summary(df_flagged: pd.DataFrame) -> pd.DataFrame:
    """Retourne un résumé des anomalies par SKU (nombre, semaines concernées).

    Args:
        df_flagged: sortie de ``detect_anomalies``.

    Returns:
        DataFrame avec colonnes ``stock_sku``, ``n_anomalies``, ``pct_anomalies``,
        ``max_z``.
    """
    rows = []
    for sku, grp in df_flagged.groupby("stock_sku"):
        anomalies = grp[grp["is_anomaly"]]
        rows.append({
            "stock_sku": sku,
            "n_anomalies": len(anomalies),
            "pct_anomalies": len(anomalies) / len(grp) if len(grp) > 0 else 0.0,
            "max_z": float(grp["z_score"].max()) if not grp["z_score"].isna().all() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("n_anomalies", ascending=False).reset_index(drop=True)


__all__ = [
    "DEFAULT_Z_THRESHOLD",
    "MIN_WEEKS_FOR_ZSCORE",
    "anomaly_summary",
    "detect_anomalies",
]
