#!/usr/bin/env python3
"""Dashboard statistiques membres Paheko : données démographiques et visualisations."""
import sys
from pathlib import Path
# Permet l'exécution directe `python paheko_stats/paheko.py` en plus de `python -m`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.mail_utils import (
    load_project_env,
    setup_memory_log_capture,
    send_email,
    build_log_footer,
    )
from dateutil.relativedelta import relativedelta
import requests
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse
import csv
import json
import logging
import os
import statistics
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

load_project_env(
    required_vars=["PAHEKO_BASE_URL", "PAHEKO_API_USER", "PAHEKO_API_PASSWORD"],
    logger=log,
    )

_log_buffer, _log_handler = setup_memory_log_capture()

PAHEKO_BASE_URL = os.getenv("PAHEKO_BASE_URL", "").rstrip("/")
PAHEKO_API_USER = os.getenv("PAHEKO_API_USER", "")
PAHEKO_API_PASSWORD = os.getenv("PAHEKO_API_PASSWORD", "")

PAHEKO_SQL_LIMIT = int(os.getenv("PAHEKO_SQL_LIMIT", "10000"))
PAHEKO_SAVE_RAW_JSON = os.getenv("PAHEKO_SAVE_RAW_JSON", "1").lower() in {"1", "true", "yes", "on"}

PAHEKO_USERS_TABLE = os.getenv("PAHEKO_USERS_TABLE", "users")
PAHEKO_CATEGORIES_TABLE = os.getenv("PAHEKO_CATEGORIES_TABLE", "users_categories")

PAHEKO_USER_CATEGORY_FIELD = os.getenv("PAHEKO_USER_CATEGORY_FIELD", "id_category")
PAHEKO_CATEGORY_ID_FIELD = os.getenv("PAHEKO_CATEGORY_ID_FIELD", "id")
PAHEKO_CATEGORY_LABEL_FIELD = os.getenv("PAHEKO_CATEGORY_LABEL_FIELD", "name")

PAHEKO_FIELD_BIRTHDATE = os.getenv("PAHEKO_FIELD_BIRTHDATE", "").strip()
PAHEKO_FIELD_BIRTHYEAR = os.getenv("PAHEKO_FIELD_BIRTHYEAR", "").strip()
PAHEKO_FIELD_NEWSLETTER = os.getenv("PAHEKO_FIELD_NEWSLETTER", "").strip()
PAHEKO_FIELD_SIGNUP_DATE = os.getenv("PAHEKO_FIELD_SIGNUP_DATE", "").strip()
PAHEKO_FIELD_POSTAL_CODE = os.getenv("PAHEKO_FIELD_POSTAL_CODE", "").strip()
PAHEKO_FIELD_CITY = os.getenv("PAHEKO_FIELD_CITY", "").strip()

session = requests.Session()
session.auth = (PAHEKO_API_USER, PAHEKO_API_PASSWORD)
session.headers.update({"Accept": "application/json"})

COLORS = {
    "primary": "#403B3A",
    "secondary": "#00818A",
    "accent": "#00818A",
    "accent2": "#FFA70B",
    "accent3": "#E05A2B",
    "accent4": "#403B3A",
    "background": "#F8FAFC",
    "card": "#FFFFFF",
    "success": "#00818A",
    "warning": "#FFA70B",
    "text": "#403B3A",
    "text_light": "#6B6564",
    "text_muted": "#9E9897",
    "border": "#E2E8F0",
    }

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.facecolor": COLORS["card"],
    "figure.facecolor": COLORS["background"],
    "axes.grid": False,
    "axes.edgecolor": COLORS["border"],
    })


def to_clean_str(value):
    """Convertit une valeur en chaîne propre (sans espaces superflus)."""
    if value is None:
        return ""

    return str(value).strip()


def qualify_sql_expr(expr, table_alias="u"):
    """Qualifie une expression SQL simple avec l'alias de table."""
    expr = to_clean_str(expr)

    if not expr:
        return ""
    special_chars = [" ", "(", ")", ".", '"', "'", ","]

    if any(ch in expr for ch in special_chars):
        return expr

    return f"{table_alias}.{expr}"


def sql_select_expr(sql_expr, alias, table_alias="u"):
    """Retourne une clause SELECT qualifiée ou NULL AS alias si vide."""
    qualified = qualify_sql_expr(sql_expr, table_alias=table_alias)

    if qualified:
        return f'{qualified} AS "{alias}"'

    return f'NULL AS "{alias}"'


def paheko_request(method, path, *, data=None, params=None):
    """Effectue une requête vers l'API Paheko et retourne le résultat JSON ou texte."""
    url = f"{PAHEKO_BASE_URL}/api/{path.lstrip('/')}"
    response = session.request(method, url, data=data, params=params, timeout=60)

    if not response.ok:
        try:
            payload = response.json()
            error_msg = payload.get("error") or payload.get("message") or response.text
        except Exception:
            error_msg = response.text
        raise RuntimeError(f"Erreur API Paheko [{response.status_code}] {path}: {error_msg}")

    content_type = response.headers.get("Content-Type", "")

    if "json" in content_type:
        return response.json()

    return response.text


def run_sql(sql):
    """Exécute une requête SQL via l'API Paheko et retourne les résultats."""
    payload = paheko_request("POST", "sql/", data={"sql": sql})

    if isinstance(payload, dict):
        return payload.get("results", [])

    if isinstance(payload, list):
        return payload

    return []


def get_table_columns(table: str) -> set:
    """Retourne l'ensemble des colonnes d'une table Paheko (avec cache local)."""
    _columns_cache: dict[str, set] = {}

    if table in _columns_cache:
        return _columns_cache[table]

    try:
        rows = run_sql(f'SELECT * FROM "{table}" LIMIT 1;')
        cols = set(rows[0].keys()) if rows else set()
    except Exception as e:
        log.warning("Impossible de lire les colonnes de %s: %s", table, e)
        cols = set()

    _columns_cache[table] = cols

    return cols


def pick_existing_key(row, candidates):
    """Retourne le premier candidat présent dans row, ou None."""
    for key in candidates:
        if key and key in row:
            return key

    return None


def normalize_newsletter(value):
    """Normalise la valeur d'abonnement newsletter en 'Oui', 'Non' ou la valeur brute."""
    value = to_clean_str(value)

    if not value:
        return "Non renseigné"

    lower = value.lower()

    if lower in {"oui", "yes", "true", "1", "on"}:
        return "Oui"

    if lower in {"non", "no", "false", "0", "off"}:
        return "Non"

    return value


def parse_date_flexible(value):
    """Parse une date depuis plusieurs formats possibles (dd/mm/yyyy, ISO, etc.)."""
    value = to_clean_str(value)

    if not value:
        return None

    value = value.split(" ", maxsplit=1)[0]
    patterns = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        ]

    for pattern in patterns:
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def month_key_from_date(value):
    """Retourne une clé mensuelle 'YYYY-MM' depuis une date, ou None."""
    dt = parse_date_flexible(value)

    if not dt:
        return None

    return dt.strftime("%Y-%m")


def find_category_count(cat_counts, expected_name):
    """Cherche le comptage d'une catégorie par son nom normalisé."""
    expected = to_clean_str(expected_name).lower()

    for name, count in cat_counts.items():
        if to_clean_str(name).lower() == expected:
            return count

    return 0


def fetch_categories_lookup():
    """Charge le dictionnaire {id → nom} des catégories depuis Paheko."""
    sql = f"SELECT * FROM {PAHEKO_CATEGORIES_TABLE} LIMIT {PAHEKO_SQL_LIMIT};"
    rows = run_sql(sql)
    category_lookup = {}

    for row in rows:
        id_key = pick_existing_key(row, [
            PAHEKO_CATEGORY_ID_FIELD,
            "id",
            "id_category",
            "category_id",
            ])
        label_key = pick_existing_key(row, [
            PAHEKO_CATEGORY_LABEL_FIELD,
            "name",
            "label",
            "title",
            "nom",
            ])

        if not id_key:
            continue

        category_id = to_clean_str(row.get(id_key))

        if not category_id:
            continue

        label = to_clean_str(row.get(label_key)) if label_key else ""
        category_lookup[category_id] = label or f"Catégorie {category_id}"

    if PAHEKO_SAVE_RAW_JSON:
        with open(BASE_DIR / "paheko_categories_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    return category_lookup


def build_members_sql():
    """Construit la requête SQL pour récupérer les membres avec les champs configurés."""
    known_cols = get_table_columns(PAHEKO_USERS_TABLE)

    def safe_expr(field_var, alias):
        expr = to_clean_str(field_var)

        if not expr:
            return f'NULL AS "{alias}"'
        # Si c'est un identifiant simple (pas une expression SQL), vérifie qu'il existe
        is_simple_ident = not any(ch in expr for ch in [" ", "(", ")", ".", '"', "'", ","])

        if is_simple_ident and known_cols and expr not in known_cols:
            log.warning(
                'Colonne configurée introuvable dans %s pour le champ "%s". '
                "Vérifiez la variable .env correspondante.",
                PAHEKO_USERS_TABLE, alias,
            )

            return f'NULL AS "{alias}"'
        qualified = qualify_sql_expr(expr, table_alias="u")

        return f'{qualified} AS "{alias}"'

    select_parts = [
        'u.id AS "_user_id"',
        safe_expr(PAHEKO_USER_CATEGORY_FIELD, "_category_id"),
        safe_expr(PAHEKO_FIELD_BIRTHDATE, "Date de naissance complète"),
        safe_expr(PAHEKO_FIELD_BIRTHYEAR, "Année de naissance"),
        safe_expr(PAHEKO_FIELD_NEWSLETTER, "Inscription à la lettre d'information"),
        safe_expr(PAHEKO_FIELD_SIGNUP_DATE, "Date d'inscription"),
        safe_expr(PAHEKO_FIELD_POSTAL_CODE, "Code postal"),
        safe_expr(PAHEKO_FIELD_CITY, "Ville"),
    ]

    return f"""
SELECT
    {", ".join(select_parts)}
FROM {PAHEKO_USERS_TABLE} u
LIMIT {PAHEKO_SQL_LIMIT};
""".strip()


def fetch_members_rows(category_lookup):
    """Récupère et normalise les lignes membres depuis Paheko."""
    sql = build_members_sql()
    rows = run_sql(sql)
    normalized_rows = []

    for row in rows:
        category_id = to_clean_str(row.get("_category_id"))
        category_name = category_lookup.get(category_id, "")

        normalized_rows.append({
            "Date de naissance complète": to_clean_str(row.get("Date de naissance complète")),
            "Année de naissance": to_clean_str(row.get("Année de naissance")),
            "Catégorie": category_name,
            "Inscription à la lettre d'information": normalize_newsletter(
                row.get("Inscription à la lettre d'information")
                ),
            "Date d'inscription": to_clean_str(row.get("Date d'inscription")),
            "Code postal": to_clean_str(row.get("Code postal")),
            "Ville": to_clean_str(row.get("Ville")),
            })

    if PAHEKO_SAVE_RAW_JSON:
        with open(BASE_DIR / "paheko_membres_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(normalized_rows, f, ensure_ascii=False, indent=2)

    return normalized_rows


def create_kpi_card(ax, value, label, color, subtext="", sample=""):
    """Dessine une carte KPI dans l'axe donné avec valeur, label et couleur."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    shadow = mpatches.FancyBboxPatch(
        (0.03, 0.02), 0.96, 0.88,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#00000008", edgecolor="none"
        )
    ax.add_patch(shadow)

    rect = mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.90,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=COLORS["card"], edgecolor=COLORS["border"], linewidth=1
        )
    ax.add_patch(rect)

    accent_bar = mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.04, 0.90,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=color, edgecolor="none"
        )
    ax.add_patch(accent_bar)

    ax.text(
        0.55, 0.58, str(value), ha="center", va="center",
        fontsize=28, fontweight="bold", color=COLORS["primary"]
        )
    ax.text(
        0.55, 0.28, label, ha="center", va="center",
        fontsize=10, color=COLORS["text_light"], fontweight="medium"
        )

    if subtext:
        ax.text(
            0.55, 0.14, subtext, ha="center", va="center",
            fontsize=8, color=COLORS["text_muted"]
            )

    if sample:
        ax.text(
            0.95, 0.08, sample, ha="right", va="bottom",
            fontsize=7, color=COLORS["text_muted"], style="italic", alpha=0.7
            )


def add_card_background(fig, rect, title=""):
    """Ajoute un fond de carte arrondi dans la figure matplotlib."""
    card_ax = fig.add_axes(rect, zorder=1)
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis("off")

    shadow = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#00000006", edgecolor="none"
        )
    card_ax.add_patch(shadow)

    rect_patch = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=COLORS["card"], edgecolor=COLORS["border"], linewidth=1
        )
    card_ax.add_patch(rect_patch)

    if title:
        card_ax.text(
            0.04, 0.92, title, ha="left", va="top",
            fontsize=12, fontweight="bold", color=COLORS["primary"]
            )

    return card_ax


def compute_stats(lignes):
    """Calcule les statistiques agrégées (âges, catégories, newsletter, etc.)."""
    date_ref = datetime.now()

    ages_precis = []
    categories = []
    inscriptions_newsletter = []
    dates_inscription = []
    codes_postaux = []
    villes = []

    for ligne in lignes:
        date_complete = to_clean_str(ligne.get("Date de naissance complète"))
        annee_existante = to_clean_str(ligne.get("Année de naissance"))

        age_calcule = None
        date_naissance = parse_date_flexible(date_complete)

        if date_naissance:
            try:
                age_calcule = relativedelta(date_ref, date_naissance).years
            except Exception:
                age_calcule = None

        if age_calcule is None and annee_existante:
            try:
                annee = int(float(annee_existante))
                age_calcule = date_ref.year - annee
            except Exception:
                pass

        if age_calcule and 0 < age_calcule < 120:
            ages_precis.append(age_calcule)

        if not annee_existante and date_naissance:
            ligne["Année de naissance"] = str(date_naissance.year)

        categorie = to_clean_str(ligne.get("Catégorie"))
        if categorie:
            categories.append(categorie)

        newsletter = normalize_newsletter(ligne.get("Inscription à la lettre d'information"))
        inscriptions_newsletter.append(newsletter)

        date_inscr = to_clean_str(ligne.get("Date d'inscription"))
        if date_inscr:
            dates_inscription.append(date_inscr)

        cp = to_clean_str(ligne.get("Code postal"))
        if cp:
            codes_postaux.append(cp)

        ville = to_clean_str(ligne.get("Ville"))
        if ville:
            villes.append(ville)

    cat_counts = Counter(categories)
    news_counts = Counter(inscriptions_newsletter)

    total_membres = len(lignes)
    abonnes_news = news_counts.get("Oui", 0)
    taux_abo = (abonnes_news / total_membres * 100) if total_membres > 0 else 0

    return {
        "total_membres": total_membres,
        "cat_counts": cat_counts,
        "news_counts": news_counts,
        "ages_precis": ages_precis,
        "dates_inscription": dates_inscription,
        "codes_postaux": codes_postaux,
        "villes": villes,
        "membres_actifs": find_category_count(cat_counts, "Membres actifs"),
        "anciens_membres": find_category_count(cat_counts, "Anciens membres"),
        "administrateurs": find_category_count(cat_counts, "Administrateurs"),
        "referents": find_category_count(cat_counts, "Référent au CAC"),
        "taux_abo": taux_abo,
        "n_ages": len(ages_precis),
        "n_categories": len(categories),
        "n_newsletter": len(inscriptions_newsletter),
        "n_inscriptions": len(dates_inscription),
        "n_codes_postaux": len(codes_postaux),
        "n_villes": len(villes),
        "n_codes_uniques": len(set(codes_postaux)),
        "n_villes_uniques": len(set(villes)),
        "age_moyen": sum(ages_precis) / len(ages_precis) if ages_precis else 0,
        "age_median": statistics.median(ages_precis) if ages_precis else 0,
        "age_min": min(ages_precis) if ages_precis else 0,
        "age_max": max(ages_precis) if ages_precis else 0,
        }


def render_dashboard(stats, output_path):
    """Génère le dashboard matplotlib et l'enregistre en PNG."""
    fig = plt.figure(figsize=(20, 13), facecolor=COLORS["background"])

    header_ax = fig.add_axes([0, 0.90, 1, 0.10])
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.axis("off")
    header_ax.set_facecolor(COLORS["primary"])
    header_ax.axhspan(0, 1, color=COLORS["primary"], zorder=1)

    header_ax.text(
        0.5, 0.65, "TABLEAU DE BORD DES MEMBRES",
        ha="center", va="center", fontsize=26, fontweight="bold",
        color="white", zorder=2
        )
    header_ax.text(
        0.5, 0.25,
        f'Analyse au {datetime.now().strftime("%d/%m/%Y")} • {stats["total_membres"]} membres au total',
        ha="center", va="center", fontsize=13, color=COLORS["text_muted"], zorder=2
        )

    ax_kpi1 = fig.add_axes([0.02, 0.78, 0.145, 0.10])
    ax_kpi2 = fig.add_axes([0.185, 0.78, 0.145, 0.10])
    ax_kpi3 = fig.add_axes([0.35, 0.78, 0.145, 0.10])
    ax_kpi4 = fig.add_axes([0.515, 0.78, 0.145, 0.10])
    ax_kpi5 = fig.add_axes([0.68, 0.78, 0.145, 0.10])
    ax_kpi6 = fig.add_axes([0.845, 0.78, 0.135, 0.10])

    create_kpi_card(ax_kpi1, stats["membres_actifs"], "Membres actifs",
                    COLORS["accent"], sample=f'n={stats["n_categories"]}')
    create_kpi_card(ax_kpi2, stats["anciens_membres"], "Anciens membres",
                    COLORS["accent3"], sample=f'n={stats["n_categories"]}')
    create_kpi_card(ax_kpi3, f'{stats["taux_abo"]:.0f}%', "Abonnés newsletter",
                    COLORS["accent2"], sample=f'n={stats["n_newsletter"]}')
    create_kpi_card(ax_kpi4, f'{stats["age_moyen"]:.0f}', "Âge moyen", COLORS["accent4"],
                    subtext=f'{stats["age_min"]} - {stats["age_max"]} ans',
                    sample=f'n={stats["n_ages"]}')
    create_kpi_card(ax_kpi5, stats["n_codes_uniques"], "Codes postaux",
                    COLORS["accent2"], sample=f'n={stats["n_codes_postaux"]}')
    create_kpi_card(ax_kpi6, stats["n_villes_uniques"], "Villes",
                    COLORS["accent"], sample=f'n={stats["n_villes"]}')

    add_card_background(fig=fig, rect=[0.02, 0.42, 0.31, 0.34], title="Répartition par catégorie")
    ax1 = fig.add_axes([0.04, 0.44, 0.27, 0.28], zorder=2)
    ax1.set_facecolor(COLORS["card"])

    palette = [
        "#00818A",
        "#FFA70B",
        "#E05A2B",
        "#6B6564",
        "#14B8A6",
        "#FFCB4F",
        "#006670",
        "#C8860A",
        "#F97316",
        "#00A3AF",
        ]

    cat_items = stats["cat_counts"].most_common()
    cat_order = [name for name, _ in cat_items]
    cat_values = [value for _, value in cat_items]
    cat_colors = [palette[i % len(palette)] for i in range(len(cat_items))]

    if cat_values:
        wedges, _, _ = ax1.pie(
            cat_values,
            labels=None,
            autopct="",
            colors=cat_colors,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": COLORS["card"], "linewidth": 2}
            )

        centre_circle = plt.Circle((0, 0), 0.38, fc=COLORS["card"])
        ax1.add_patch(centre_circle)

        ax1.text(0, 0.05, f"{sum(cat_values)}", ha="center", va="center",
                 fontsize=22, fontweight="bold", color=COLORS["primary"])
        ax1.text(0, -0.12, "membres", ha="center", va="center",
                 fontsize=9, color=COLORS["text_light"])

        legend_labels = [f"{cat} ({val})" for cat, val in zip(cat_order, cat_values)]
        ax1.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=False,
            fontsize=9,
            labelcolor=COLORS["text"]
            )
    else:
        ax1.text(0.5, 0.5, "Aucune donnée catégorie", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=11, color=COLORS["text_light"])

    ax1.text(
        0.98, 0.02, f'n={stats["n_categories"]}', transform=ax1.transAxes,
        ha="right", va="bottom", fontsize=7, color=COLORS["text_muted"],
        style="italic", alpha=0.8
        )

    add_card_background(fig=fig, rect=[0.35, 0.42, 0.30, 0.34], title="Distribution des âges")
    ax2 = fig.add_axes([0.38, 0.44, 0.24, 0.27], zorder=2)
    ax2.set_facecolor(COLORS["card"])

    if stats["ages_precis"]:
        ax2.hist(
            stats["ages_precis"], bins=12, color=COLORS["accent2"],
            edgecolor=COLORS["card"], alpha=0.85, linewidth=1.5
            )
        ax2.axvline(stats["age_moyen"], color=COLORS["accent3"], linestyle="--", linewidth=2.5,
                    label=f'Moyenne : {stats["age_moyen"]:.0f} ans')
        ax2.axvline(stats["age_median"], color=COLORS["accent4"], linestyle=":", linewidth=2.5,
                    label=f'Médiane : {stats["age_median"]:.0f} ans')

        q1, q3 = np.percentile(stats["ages_precis"], [25, 75])
        ax2.axvspan(q1, q3, alpha=0.12, color=COLORS["accent4"],
                    label=f"50% : {int(q1)}-{int(q3)} ans")

        ax2.set_xlabel("Âge", color=COLORS["text_light"], fontsize=9)
        ax2.set_ylabel("Membres", color=COLORS["text_light"], fontsize=9)
        ax2.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=COLORS["text"])
        ax2.tick_params(colors=COLORS["text_light"], labelsize=8)
    else:
        ax2.text(0.5, 0.5, "Aucune donnée d'âge exploitable", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=11, color=COLORS["text_light"])
        ax2.set_xticks([])
        ax2.set_yticks([])

    ax2.text(
        0.98, 0.02, f'n={stats["n_ages"]}', transform=ax2.transAxes,
        ha="right", va="top", fontsize=7, color=COLORS["text_muted"],
        style="italic", alpha=0.8
        )

    add_card_background(fig=fig, rect=[0.67, 0.42, 0.31, 0.34], title="Inscription newsletter")
    ax3 = fig.add_axes([0.70, 0.44, 0.25, 0.27], zorder=2)
    ax3.set_facecolor(COLORS["card"])

    news_labels = list(stats["news_counts"].keys())
    news_values = list(stats["news_counts"].values())
    news_colors = [
        COLORS["success"] if "Oui" in label
        else COLORS["accent3"] if "Non" in label
        else COLORS["secondary"]
        for label in news_labels
        ]

    if news_values:
        bars = ax3.barh(
            news_labels, news_values, color=news_colors, height=0.55,
            edgecolor=COLORS["card"], linewidth=2
            )

        for rect_bar, val in zip(bars, news_values):
            ax3.text(
                rect_bar.get_width() + 1, rect_bar.get_y() + rect_bar.get_height() / 2,
                f"{val}", va="center", fontsize=11, fontweight="bold", color=COLORS["primary"],
            )

        ax3.set_xlabel("Membres", color=COLORS["text_light"], fontsize=9)
        ax3.tick_params(colors=COLORS["text_light"], labelsize=9)
        ax3.set_xlim(0, max(news_values) * 1.25 if max(news_values) > 0 else 1)
    else:
        ax3.text(0.5, 0.5, "Aucune donnée newsletter", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=11, color=COLORS["text_light"])
        ax3.set_xticks([])
        ax3.set_yticks([])

    ax3.text(
        0.98, 0.02, f'n={stats["n_newsletter"]}', transform=ax3.transAxes,
        ha="right", va="bottom", fontsize=7, color=COLORS["text_muted"],
        style="italic", alpha=0.8
        )

    add_card_background(fig=fig, rect=[0.02, 0.04, 0.46, 0.34], title="Évolution des inscriptions")
    ax4 = fig.add_axes([0.05, 0.06, 0.40, 0.27], zorder=2)
    ax4.set_facecolor(COLORS["card"])

    mois_inscriptions = []
    for date in stats["dates_inscription"]:
        key = month_key_from_date(date)
        if key:
            mois_inscriptions.append(key)

    if mois_inscriptions:
        mois_counts = Counter(mois_inscriptions)
        mois_tries = sorted(mois_counts.keys())
        values = [mois_counts[m] for m in mois_tries]

        ax4.fill_between(range(len(mois_tries)), values, alpha=0.2, color=COLORS["accent"])
        ax4.plot(
            range(len(mois_tries)), values, marker="o", color=COLORS["accent"],
            linewidth=3, markersize=7, markerfacecolor=COLORS["card"], markeredgewidth=2
            )

        max_idx = values.index(max(values))
        ax4.annotate(
            f"{max(values)}", (max_idx, max(values)),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10, fontweight="bold", color=COLORS["primary"],
            bbox={"boxstyle": "round,pad=0.3", "facecolor": COLORS["accent"], "alpha": 0.2}
            )

        ax4.set_xticks(range(len(mois_tries)))
        ax4.set_xticklabels(
            [datetime.strptime(m, "%Y-%m").strftime("%m/%y") for m in mois_tries],
            rotation=45, ha="right", fontsize=8
            )
        ax4.set_xlabel("Mois", color=COLORS["text_light"], fontsize=9)
        ax4.set_ylabel("Inscriptions", color=COLORS["text_light"], fontsize=9)
        ax4.tick_params(colors=COLORS["text_light"], labelsize=8)
        ax4.set_xlim(-0.5, len(mois_tries) - 0.5)
        ax4.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    else:
        ax4.text(0.5, 0.5, "Aucune donnée d'inscription", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=11, color=COLORS["text_light"])
        ax4.set_xticks([])
        ax4.set_yticks([])

    ax4.text(
        0.98, 0.98, f'n={stats["n_inscriptions"]}', transform=ax4.transAxes,
        ha="right", va="top", fontsize=7, color=COLORS["text_muted"],
        style="italic", alpha=0.8
        )

    add_card_background(fig=fig, rect=[0.50, 0.04, 0.48, 0.34], title="")
    top_villes = Counter(stats["villes"]).most_common(10)
    top_cp = Counter(stats["codes_postaux"]).most_common(10)

    ax_villes = fig.add_axes([0.52, 0.06, 0.21, 0.27], zorder=2)
    ax_villes.set_facecolor(COLORS["card"])

    ax_cp = fig.add_axes([0.75, 0.06, 0.21, 0.27], zorder=2)
    ax_cp.set_facecolor(COLORS["card"])

    if top_villes:
        labels_villes = [v[0][:14] + "..." if len(v[0]) > 14 else v[0] for v in top_villes]
        values_villes = [v[1] for v in top_villes]
        y_pos_villes = list(range(len(top_villes)))

        bars_v = ax_villes.barh(
            y_pos_villes, values_villes, color=COLORS["accent"],
            height=0.6, edgecolor=COLORS["card"], linewidth=1.5
            )

        for rect_bar, val in zip(bars_v, values_villes):
            ax_villes.text(
                rect_bar.get_width() + 1, rect_bar.get_y() + rect_bar.get_height() / 2,
                f"{val}", va="center", fontsize=9, fontweight="bold", color=COLORS["primary"],
            )

        ax_villes.set_yticks(y_pos_villes)
        ax_villes.set_yticklabels(labels_villes, fontsize=9)
        ax_villes.invert_yaxis()
        ax_villes.set_xlabel("Membres", fontsize=9, color=COLORS["text_light"])
        ax_villes.tick_params(colors=COLORS["text_light"], bottom=False, labelsize=8)
        ax_villes.set_xlim(0, max(values_villes) * 1.35 if max(values_villes) > 0 else 1)

        ax_villes.text(0.5, 1.02, "Top 10 Villes", ha="center", va="bottom", fontsize=10,
                       fontweight="bold", color=COLORS["text"], transform=ax_villes.transAxes)
    else:
        ax_villes.text(0.5, 0.5, "Aucune donnée ville", transform=ax_villes.transAxes,
                       ha="center", va="center", fontsize=11, color=COLORS["text_light"])
        ax_villes.set_xticks([])
        ax_villes.set_yticks([])

    ax_villes.text(
        0.98, 0.02, f'n={stats["n_villes"]}', transform=ax_villes.transAxes,
        ha="right", va="bottom", fontsize=7, color=COLORS["text_muted"],
        style="italic", alpha=0.8
        )

    if top_cp:
        labels_cp = [cp[0] for cp in top_cp]
        values_cp = [cp[1] for cp in top_cp]
        y_pos_cp = list(range(len(top_cp)))

        bars_c = ax_cp.barh(
            y_pos_cp, values_cp, color=COLORS["accent2"],
            height=0.6, edgecolor=COLORS["card"], linewidth=1.5
            )

        for rect_bar, val in zip(bars_c, values_cp):
            ax_cp.text(
                rect_bar.get_width() + 1, rect_bar.get_y() + rect_bar.get_height() / 2,
                f"{val}", va="center", fontsize=9, fontweight="bold", color=COLORS["primary"],
            )

        ax_cp.set_yticks(y_pos_cp)
        ax_cp.set_yticklabels(labels_cp, fontsize=9)
        ax_cp.invert_yaxis()
        ax_cp.set_xlabel("Membres", fontsize=9, color=COLORS["text_light"])
        ax_cp.tick_params(colors=COLORS["text_light"], bottom=False, labelsize=8)
        ax_cp.set_xlim(0, max(values_cp) * 1.35 if max(values_cp) > 0 else 1)

        ax_cp.text(0.5, 1.02, "Top 10 Codes postaux", ha="center", va="bottom", fontsize=10,
                   fontweight="bold", color=COLORS["text"], transform=ax_cp.transAxes)
    else:
        ax_cp.text(0.5, 0.5, "Aucune donnée code postal", transform=ax_cp.transAxes,
                   ha="center", va="center", fontsize=11, color=COLORS["text_light"])
        ax_cp.set_xticks([])
        ax_cp.set_yticks([])

    ax_cp.text(
        0.98, 0.02, f'n={stats["n_codes_postaux"]}', transform=ax_cp.transAxes,
        ha="right", va="bottom", fontsize=7, color=COLORS["text_muted"],
        style="italic", alpha=0.8
        )

    ax_sep = fig.add_axes([0.74, 0.06, 0.005, 0.27], zorder=2)
    ax_sep.axvline(0.5, color=COLORS["border"], linewidth=1, alpha=0.5)
    ax_sep.set_xlim(0, 1)
    ax_sep.set_ylim(0, 1)
    ax_sep.axis("off")

    for ax in [ax1, ax2, ax3, ax4, ax_villes, ax_cp]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=COLORS["background"],
        edgecolor="none",
        )
    plt.close(fig)


def save_anonymized_csv(lignes, output_path):
    """Exporte les données membres anonymisées (sans date de naissance) en CSV."""
    fieldnames = [
        "Date de naissance complète",
        "Année de naissance",
        "Catégorie",
        "Inscription à la lettre d'information",
        "Date d'inscription",
        "Code postal",
        "Ville",
        ]

    lignes_csv = [dict(ligne) for ligne in lignes]
    for ligne in lignes_csv:
        ligne["Date de naissance complète"] = ""

    with open(output_path, "w", encoding="utf-8", newline="") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        writer.writerows(lignes_csv)


def send_dashboard_email(stats, dashboard_path, csv_path):
    """Envoie le dashboard PNG et le CSV par email avec un résumé des statistiques."""
    today = datetime.now().strftime("%d/%m/%Y")
    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    logs_str = build_log_footer(_log_buffer)

    attachments = [dashboard_path, csv_path]
    members_json = BASE_DIR / "paheko_membres_snapshot.json"
    categories_json = BASE_DIR / "paheko_categories_snapshot.json"

    if members_json.exists():
        attachments.append(members_json)
    if categories_json.exists():
        attachments.append(categories_json)

    subject = (
        f'Dashboard membres Paheko — {today} '
        f'({stats["total_membres"]} membres, {stats["membres_actifs"]} actifs)'
        )

    body = f"""\
Bonjour,

Veuillez trouver ci-joint le tableau de bord des membres Paheko généré automatiquement.

Date d'analyse : {today}
Total membres : {stats["total_membres"]}
Membres actifs : {stats["membres_actifs"]}
Anciens membres : {stats["anciens_membres"]}
Administrateurs : {stats["administrateurs"]}
Référent au CAC : {stats["referents"]}
Abonnés newsletter : {stats["taux_abo"]:.0f}%
Âge moyen : {stats["age_moyen"]:.0f} ans
Codes postaux uniques : {stats["n_codes_uniques"]}
Villes uniques : {stats["n_villes_uniques"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JOURNAL D'EXÉCUTION — généré le {now_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{logs_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cordialement,
Corentin via {Path(__file__).name}
    """

    send_email(
        subject=subject,
        body=body,
        attachments=attachments,
        mailing_list="all_ca",
        logger=log,
        )


def run_dashboard(send_mail: bool = True):
    """Pipeline complet : récupère, calcule, génère le dashboard et envoie l'email."""
    log.info("Étape 1/5 - Récupération des catégories…")
    category_lookup = fetch_categories_lookup()

    log.info("Étape 2/5 - Récupération des membres…")
    lignes = fetch_members_rows(category_lookup)

    log.info("Étape 3/5 - Calcul des statistiques…")
    stats = compute_stats(lignes)

    dashboard_path = BASE_DIR / "statistiques_membres_dashboard.png"
    csv_path = BASE_DIR / "membres_anonymises.csv"

    log.info("Étape 4/5 - Génération du dashboard…")
    render_dashboard(stats, dashboard_path)
    save_anonymized_csv(lignes, csv_path)

    log.info("Dashboard cree : %s", dashboard_path.name)
    log.info("CSV anonymise cree : %s", csv_path.name)

    if send_mail:
        log.info("Étape 5/5 - Envoi par email…")
        send_dashboard_email(stats, dashboard_path, csv_path)
    else:
        log.info("Étape 5/5 - Envoi email ignoré (--no-mail).")

    log.info("Terminé.")


def main():
    """Point d'entrée CLI pour le dashboard Paheko."""
    parser = argparse.ArgumentParser(description="Dashboard membres Paheko")
    parser.add_argument(
        "--no-mail",
        action="store_true",
        help="Génère le dashboard et les fichiers sans envoyer d'email",
        )
    args = parser.parse_args()
    run_dashboard(send_mail=not args.no_mail)


if __name__ == "__main__":
    main()
