#!/usr/bin/env python3
"""Persistance de l'historique hebdomadaire de consommation par SKU.

Schéma du parquet :
    stock_sku    str    SKU de stock
    week_label   str    'YYYY-Www' ISO (clé lisible)
    year         int    année ISO
    week         int    numéro de semaine ISO
    week_start   date   lundi de la semaine
    usage        float  quantité consommée (avec consumption_per_sale)
    sales_count  int    nombre de ventes brutes (sans facteur)

La clé unique est (stock_sku, week_label). Les écritures sont idempotentes :
appeler `update_weekly_usage` avec une ligne déjà présente met à jour les
valeurs (pas de doublon créé), ce qui permet de relancer le rapport sans
corrompre l'historique.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from utils.sumup_shared import week_start as iso_week_start

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "weekly_usage.parquet"

SCHEMA_COLUMNS = [
    "stock_sku",
    "week_label",
    "year",
    "week",
    "week_start",
    "usage",
    "sales_count",
]


@dataclass(frozen=True)
class WeeklyRow:
    stock_sku: str
    week_label: str
    usage: float
    sales_count: int

    @property
    def year(self) -> int:
        return int(self.week_label.split("-W")[0])

    @property
    def week(self) -> int:
        return int(self.week_label.split("-W")[1])

    @property
    def week_start(self) -> date:
        return iso_week_start(self.year, self.week)


def _empty_dataframe() -> pd.DataFrame:
    df = pd.DataFrame({col: pd.Series(dtype=_dtype_for(col)) for col in SCHEMA_COLUMNS})
    return df


def _dtype_for(col: str) -> str:
    return {
        "stock_sku": "string",
        "week_label": "string",
        "year": "int32",
        "week": "int32",
        "week_start": "object",
        "usage": "float64",
        "sales_count": "int64",
    }[col]


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les colonnes / dtypes sur le schéma cible."""
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype=_dtype_for(col))
    df = df[SCHEMA_COLUMNS].copy()
    df["stock_sku"] = df["stock_sku"].astype("string")
    df["week_label"] = df["week_label"].astype("string")
    df["year"] = df["year"].astype("int32")
    df["week"] = df["week"].astype("int32")
    df["usage"] = df["usage"].astype("float64")
    df["sales_count"] = df["sales_count"].astype("int64")
    df["week_start"] = df["week_start"].apply(_to_date)
    return df


def _to_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"valeur date invalide : {value!r}")


def _parse_week_label(week_label: str) -> tuple[int, int]:
    year_str, week_str = week_label.split("-W")
    return int(year_str), int(week_str)


def load_weekly_usage(path: Path | None = None) -> pd.DataFrame:
    """Charge l'historique hebdo persistant. Retourne un DataFrame vide si absent."""
    target = Path(path) if path is not None else DATASET_PATH
    if not target.exists():
        return _empty_dataframe()
    df = pd.read_parquet(target)
    return _coerce_dtypes(df)


def save_weekly_usage(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Écrit l'historique hebdo en parquet (création répertoire si besoin)."""
    target = Path(path) if path is not None else DATASET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df = _coerce_dtypes(df)
    df = df.sort_values(["stock_sku", "year", "week"]).reset_index(drop=True)
    df.to_parquet(target, index=False)
    return target


def weekly_usage_dict_to_dataframe(
    weekly_usage: Mapping[str, Mapping[str, float]],
    weekly_sales_count: Mapping[str, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    """Convertit la structure dict produite par `aggregate_weekly_stock_usage`.

    weekly_usage      : {sku: {week_label: usage_qty}}
    weekly_sales_count: {sku: {week_label: nb_ventes}} (optionnel)
    """
    rows: list[dict] = []
    sales_lookup = weekly_sales_count or {}
    for sku, by_week in weekly_usage.items():
        sku_sales = sales_lookup.get(sku, {})
        for week_label, usage in by_week.items():
            year, week = _parse_week_label(week_label)
            rows.append(
                {
                    "stock_sku": sku,
                    "week_label": week_label,
                    "year": year,
                    "week": week,
                    "week_start": iso_week_start(year, week),
                    "usage": float(usage or 0.0),
                    "sales_count": int(sku_sales.get(week_label, 0) or 0),
                }
            )
    if not rows:
        return _empty_dataframe()
    return _coerce_dtypes(pd.DataFrame(rows))


def merge_weekly_usage(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Fusion idempotente : les lignes de `new` écrasent celles de `existing`
    pour la même clé (stock_sku, week_label).
    """
    if existing is None or len(existing) == 0:
        return _coerce_dtypes(new.copy())
    if new is None or len(new) == 0:
        return _coerce_dtypes(existing.copy())
    existing = _coerce_dtypes(existing)
    new = _coerce_dtypes(new)
    new_keys = set(zip(new["stock_sku"].tolist(), new["week_label"].tolist()))
    keep_mask = [
        (sku, wl) not in new_keys
        for sku, wl in zip(existing["stock_sku"].tolist(), existing["week_label"].tolist())
    ]
    kept = existing.loc[keep_mask]
    merged = pd.concat([kept, new], ignore_index=True)
    merged = merged.sort_values(["stock_sku", "year", "week"]).reset_index(drop=True)
    return merged


def update_weekly_usage(
    weekly_usage: Mapping[str, Mapping[str, float]],
    weekly_sales_count: Mapping[str, Mapping[str, int]] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Met à jour le parquet sur disque à partir des dicts hebdo.

    Retourne le DataFrame complet après fusion. Idempotent : relancer la même
    semaine avec des valeurs corrigées remplace simplement les lignes
    correspondantes.
    """
    new_df = weekly_usage_dict_to_dataframe(weekly_usage, weekly_sales_count)
    existing = load_weekly_usage(path)
    merged = merge_weekly_usage(existing, new_df)
    save_weekly_usage(merged, path)
    return merged


def filter_skus(df: pd.DataFrame, skus: Iterable[str]) -> pd.DataFrame:
    """Restreint le DataFrame à un sous-ensemble de SKU."""
    sku_set = set(skus)
    return df[df["stock_sku"].isin(sku_set)].reset_index(drop=True)
