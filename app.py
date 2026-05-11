"""Tableau de bord — Café Associatif Le Village."""
import json
import os
import sys
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Le Village — Tableau de bord",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── brand CSS ─────────────────────────────────────────────────────────────────

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2"
    "?family=Nunito:wght@400;600;700;800;900"
    "&family=Caveat:wght@500;600&display=swap');"
)

st.markdown(
    "<style>\n" + _FONT_IMPORT + """

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Nunito', sans-serif !important;
}

/* headers */
h1 { color: #403B3A !important; font-weight: 900 !important; font-size: 2rem !important; }
h2 { color: #00818A !important; font-weight: 800 !important; }
h3 { color: #403B3A !important; font-weight: 700 !important; }

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #403B3A !important;
    border-right: 3px solid #FFA70B !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio > label {
    color: #FFA70B !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: #FFA70B !important;
    border: 1px solid #FFA70B !important;
    font-size: 0.85rem !important;
    padding: 0.3rem 1rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #FFA70B22 !important;
}

/* primary buttons */
.stButton > button {
    background-color: #FFA70B !important;
    color: #403B3A !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.5rem 2rem !important;
    transition: background-color 0.15s ease !important;
}
.stButton > button:hover { background-color: #e6960a !important; }
.stButton > button:disabled {
    background-color: #cccccc !important;
    color: #888888 !important;
}

/* description block */
.section-intro {
    background-color: #00818A12;
    border-left: 4px solid #00818A;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.5rem;
    margin: 1rem 0 1.5rem 0;
}
.section-intro p { color: #403B3A; margin: 0; line-height: 1.7; font-size: 0.97rem; }

/* info pill */
.info-pill {
    display: inline-block;
    background-color: #FFA70B22;
    color: #403B3A;
    border: 1px solid #FFA70B;
    border-radius: 20px;
    padding: 0.15rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

/* tabs */
[data-testid="stTabs"] button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    color: #403B3A !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00818A !important;
    border-bottom-color: #00818A !important;
}

/* dividers */
hr { border-color: #FFA70B55 !important; margin: 1.25rem 0 !important; }

/* checkboxes & labels */
label { color: #403B3A !important; font-weight: 600 !important; }

/* script output */
.stCodeBlock { border-left: 3px solid #FFA70B !important; }

/* login */
.login-wrap { max-width: 420px; margin: 0 auto; text-align: center; padding-top: 2rem; }
.login-wrap .stTextInput label { font-size: 1rem !important; }
.login-tagline {
    font-family: 'Caveat', cursive !important;
    font-size: 1.3rem !important;
    color: #00818A !important;
    margin-bottom: 1.5rem !important;
    display: block;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── authentication ────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    logo_teal = Path(BASE_DIR) / "assets" / "logo_village_vert.png"
    logo_orange = Path(BASE_DIR) / "assets" / "logo_village_orange.png"
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        login_logo = logo_teal if logo_teal.exists() else logo_orange
        if login_logo.exists():
            st.image(str(login_logo), use_container_width=True)
        st.markdown("### Tableau de bord")
        st.markdown("---")
        pwd = st.text_input("Mot de passe", type="password", placeholder="Saisir le mot de passe…")
        if st.button("Se connecter", use_container_width=True):
            if pwd == st.secrets["APP_PASSWORD"]:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ── configuration des modules ─────────────────────────────────────────────────

SCRIPTS: dict[str, dict] = {
    "stocks": {
        "title": "Stocks",
        "icon": "📦",
        "nav_label": "📦  Stocks",
        "caption": "Inventaire hebdomadaire et alertes de réapprovisionnement",
        "description": (
            "Ce module récupère les transactions SumUp des dernières semaines, "
            "déduit les quantités vendues de chaque article, et génère un rapport PDF "
            "avec l'état des stocks actuels, les seuils de réapprovisionnement et les alertes. "
            "Il est recommandé d'intégrer d'abord les derniers achats (étape 1) "
            "pour que les quantités soient à jour avant de générer le rapport."
        ),
        "path": "stocks/sumup_stocks.py",
        "email_env_var": "EMAIL_TO_SUMUP_ALL_CA",
    },
    "adhesions": {
        "title": "Adhésions",
        "icon": "👥",
        "nav_label": "👥  Adhésions",
        "caption": "Transactions d'adhésion et dons SumUp",
        "description": (
            "Extrait les transactions SumUp contenant des mots-clés définis "
            "(ex : « adhésion », « don ») sur une période donnée. "
            "Génère un rapport PDF récapitulatif groupé par moyen de paiement "
            "(espèces, carte Visa, Mastercard…) avec totaux par section. "
            "Idéal pour les bilans de campagnes d'adhésion ou les rapports financiers."
        ),
        "path": "adhesions/sumup_adhesions.py",
        "email_env_var": "EMAIL_TO_SUMUP_FINANCE",
    },
    "paheko": {
        "title": "Membres",
        "icon": "🏘️",
        "nav_label": "🏘️  Membres",
        "caption": "Statistiques et démographie — Paheko",
        "description": (
            "Se connecte à l'API Paheko pour récupérer les données des membres actifs "
            "et génère un tableau de bord visuel complet : répartition par catégorie d'adhésion, "
            "pyramide des âges, taux d'abonnement newsletter, villes représentées, "
            "et évolution des inscriptions dans le temps. "
            "Utile pour préparer les assemblées générales et les bilans annuels."
        ),
        "path": "paheko_stats/paheko.py",
        "email_env_var": "EMAIL_TO_SUMUP_ALL_CA",
    },
    "stats": {
        "title": "Statistiques",
        "icon": "📊",
        "nav_label": "📊  Statistiques",
        "caption": "Analyse des ventes et performance produits",
        "description": (
            "Analyse les transactions SumUp sur une période configurable pour produire "
            "un rapport détaillé : chiffre d'affaires par produit et par catégorie, "
            "répartition des moyens de paiement, et évolution hebdomadaire des ventes. "
            "Identifie aussi les produits sans correspondance dans le catalogue "
            "(anomalies de données à corriger)."
        ),
        "path": "stocks/sumup_statistics.py",
        "email_env_var": "EMAIL_TO",
    },
}

_PURCH_ID = "purchases"

for _sid in list(SCRIPTS) + [_PURCH_ID]:
    st.session_state.setdefault(f"logs_{_sid}", [])
    st.session_state.setdefault(f"running_{_sid}", False)
    st.session_state.setdefault(f"rc_{_sid}", None)


# ── utility functions ─────────────────────────────────────────────────────────

def build_env(email_overrides=None):
    """Construit l'environnement du sous-processus depuis os.environ et st.secrets."""
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    for key, val in st.secrets.items():
        if isinstance(val, (str, int, float)):
            env[key] = str(val)
        elif hasattr(val, "items"):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, (str, int, float)):
                    env[sub_key] = str(sub_val)
    if "SUMUP_TOKEN" in st.secrets:
        env["SUMUP_API_KEY"] = str(st.secrets["SUMUP_TOKEN"])
    if "EMAIL_ADDRESS" in st.secrets:
        env.setdefault("SMTP_USER", str(st.secrets["EMAIL_ADDRESS"]))
        env.setdefault("EMAIL_FROM", str(st.secrets["EMAIL_ADDRESS"]))
    if "EMAIL_PASSWORD" in st.secrets:
        env.setdefault("SMTP_PASS", str(st.secrets["EMAIL_PASSWORD"]))
    if email_overrides:
        env.update(email_overrides)
    return env


def _default_recipients(env_var: str) -> str:
    return st.secrets.get(env_var, "")


def _sanitize_mock_file(raw_value: str) -> str:
    """Valide un chemin de fichier mock utilisateur et retourne une valeur sûre."""
    value = (raw_value or "").strip()
    if not value:
        return ""
    p = Path(value)
    if p.is_absolute():
        raise ValueError("Le fichier mock doit être un chemin relatif.")
    if ".." in p.parts:
        raise ValueError("Le fichier mock ne doit pas contenir de '..'.")
    if p.suffix.lower() != ".json":
        raise ValueError("Le fichier mock doit être un fichier .json.")
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/")
    if any(ch not in allowed_chars for ch in value):
        raise ValueError("Le fichier mock contient des caractères non autorisés.")
    return value


def _sanitize_filter_tokens(raw_value: str) -> list[str]:
    """Valide les mots-clés de filtre saisis par l'utilisateur."""
    value = raw_value or ""
    parts = value.split()
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    for tok in parts:
        if len(tok) > 50:
            raise ValueError("Un mot-clé de filtre est trop long.")
        if any(ch not in allowed_chars for ch in tok):
            raise ValueError("Les mots-clés contiennent des caractères non autorisés.")
    return parts


def build_cmd(script_cfg: dict, cmd_args: list[str]) -> list[str]:
    """Retourne la commande Python pour exécuter un script comme module."""
    module = Path(script_cfg["path"]).with_suffix("").as_posix().replace("/", ".")
    return [sys.executable, "-m", module] + cmd_args


def run_script(
    script_id: str,
    script_cmd: list[str],
    mail_env_var: str | None = None,
    email_override: str | None = None,
    extra_env: dict | None = None,
) -> None:
    """Lance un script en sous-processus et affiche sa sortie en temps réel."""
    st.session_state[f"logs_{script_id}"] = []
    st.session_state[f"rc_{script_id}"] = None
    st.session_state[f"running_{script_id}"] = True
    st.session_state[f"fresh_run_{script_id}"] = True

    overrides: dict = {}
    if mail_env_var and email_override and email_override.strip():
        overrides[mail_env_var] = email_override.strip()
    if extra_env:
        overrides.update(extra_env)

    env = build_env(email_overrides=overrides)
    log_area = st.empty()
    log_lines: list[str] = []

    with st.spinner("Script en cours…"):
        with subprocess.Popen(
            script_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=BASE_DIR,
        ) as process:
            for line in process.stdout:
                log_lines.append(line)
                log_area.code("".join(log_lines))
            process.wait()

    log_area.code("".join(log_lines))
    st.session_state[f"logs_{script_id}"] = log_lines
    st.session_state[f"rc_{script_id}"] = process.returncode
    st.session_state[f"running_{script_id}"] = False


def _show_result(
    script_id: str, skip_email: bool = False, success_label: str = "Rapport"
) -> None:
    """Affiche les logs et le statut après exécution."""
    logs = st.session_state[f"logs_{script_id}"]
    rc = st.session_state[f"rc_{script_id}"]
    if logs and not st.session_state.pop(f"fresh_run_{script_id}", False):
        st.code("".join(logs))
    if rc is not None:
        if rc == 0:
            if skip_email:
                msg = f"{success_label} généré avec succès (sans envoi email)."
            else:
                msg = f"{success_label} généré et email envoyé."
            st.success(msg)
        else:
            st.error("Une erreur s'est produite — consultez les logs ci-dessus.")


# ── sidebar ───────────────────────────────────────────────────────────────────

logo_orange = Path(BASE_DIR) / "assets" / "logo_village_orange.png"

with st.sidebar:
    if logo_orange.exists():
        st.image(str(logo_orange), use_container_width=True)
    else:
        st.markdown("### ☕ Le Village")

    st.markdown("---")

    nav_options = [cfg["nav_label"] for cfg in SCRIPTS.values()]
    nav_keys = list(SCRIPTS.keys())

    selected_nav = st.radio(
        "Module",
        options=nav_options,
        label_visibility="visible",
    )
    active_sid = nav_keys[nav_options.index(selected_nav)]

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.75rem;color:#aaa;">Café Associatif Le Village</span>',
        unsafe_allow_html=True,
    )
    if st.button("Déconnexion", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ── page content ──────────────────────────────────────────────────────────────

cfg = SCRIPTS[active_sid]
sid = active_sid
is_running = st.session_state[f"running_{sid}"]
email_env_var = cfg.get("email_env_var", "")

st.markdown(f"# {cfg['icon']}  {cfg['title']}")
st.markdown(f"*{cfg['caption']}*")
st.markdown(
    f'<div class="section-intro"><p>{cfg["description"]}</p></div>',
    unsafe_allow_html=True,
)

# ── Stocks ────────────────────────────────────────────────────────────────────

if sid == "stocks":
    tab1, tab2 = st.tabs(["📥  Étape 1 — Intégrer les achats", "📋  Étape 2 — Générer le rapport"])

    with tab1:
        st.markdown(
            "Téléchargez les dernières entrées de stock depuis Google Drive "
            "avant de générer le rapport, afin que les quantités disponibles soient à jour."
        )
        with st.expander("ℹ️  Comment fonctionne cette étape ?"):
            st.markdown(
                "Le fichier `ACHATS_suivi_stock.xlsx` est récupéré depuis Google Drive. "
                "Les colonnes d'achat sont parsées et les quantités achetées sont ajoutées "
                "au champ `stock_on_hand` de chaque article dans `stock_items.json`. "
                "Les achats déjà intégrés sont ignorés automatiquement (déduplication par date)."
            )

        _purch_running = st.session_state[f"running_{_PURCH_ID}"]
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            _dry_run = st.checkbox(
                "Mode simulation",
                value=False,
                key="dry_run_purchases",
                disabled=_purch_running,
                help="Affiche les mises à jour prévues sans modifier stock_items.json.",
            )
        with col_p2:
            _local_file = st.file_uploader(
                "Fichier Excel local (optionnel)",
                type=["xlsx"],
                key="local_xlsx_purchases",
                disabled=_purch_running,
                help="Si renseigné, utilise ce fichier au lieu de Google Drive.",
            )

        if st.button("Mettre à jour les achats", key=f"btn_{_PURCH_ID}", disabled=_purch_running):
            extra_purch_args: list[str] = []
            extra_purch_env: dict[str, str] = {}
            if _dry_run:
                extra_purch_args.append("--dry-run")
            if _local_file is not None:
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    tmp.write(_local_file.read())
                    tmp_path = tmp.name
                extra_purch_args += ["--local", tmp_path]
            if _local_file is None:
                sa_info = st.secrets.get("GDRIVE_SERVICE_ACCOUNT")
                if sa_info:
                    extra_purch_env["GDRIVE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(sa_info))
            purchases_cmd = [sys.executable, "-m", "stocks.update_stock_from_purchases"] + extra_purch_args
            run_script(_PURCH_ID, purchases_cmd, extra_env=extra_purch_env)

        _purch_logs = st.session_state[f"logs_{_PURCH_ID}"]
        _purch_rc = st.session_state[f"rc_{_PURCH_ID}"]
        if _purch_logs and not st.session_state.pop(f"fresh_run_{_PURCH_ID}", False):
            st.code("".join(_purch_logs))
        if _purch_rc is not None:
            if _purch_rc == 0:
                if st.session_state.get("dry_run_purchases", True):
                    st.info("Simulation terminée — aucune modification effectuée.")
                else:
                    st.success("stock_items.json mis à jour avec les nouvelles entrées.")
            else:
                st.error("Une erreur s'est produite — consultez les logs ci-dessus.")

    with tab2:
        st.markdown(
            "Configurez les paramètres ci-dessous, puis lancez la génération du rapport de stocks."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            weeks = st.number_input(
                "Semaines d'historique",
                min_value=1, max_value=52, value=8,
                key=f"weeks_{sid}",
                disabled=is_running,
                help=(
                    "Nombre de semaines de transactions récupérées depuis l'API SumUp "
                    "pour calculer la consommation de chaque article."
                ),
            )
        with col2:
            mock_file = st.text_input(
                "Fichier mock (optionnel)",
                value="",
                key=f"mock_{sid}",
                disabled=is_running,
                placeholder="ex : stocks/mock_transactions.json",
                help=(
                    "Chemin relatif vers un fichier JSON de transactions fictives. "
                    "Si renseigné, l'API SumUp n'est pas appelée — utile pour les tests."
                ),
            )

        no_mail = st.checkbox(
            "Ne pas envoyer l'email (générer le PDF uniquement)",
            value=False,
            key=f"no_mail_{sid}",
            disabled=is_running,
        )

        use_ml = st.checkbox(
            "Activer les projections ML (apprentissage automatique)",
            value=True,
            key=f"use_ml_{sid}",
            disabled=is_running,
            help=(
                "Ajoute au rapport des projections de rupture de stock basées sur un "
                "modèle d'apprentissage automatique entraîné sur l'historique de consommation."
            ),
        )
        if use_ml:
            st.info(
                "**Projections ML activées** — Le modèle s'entraîne progressivement chaque semaine "
                "à partir de l'historique de consommation enregistré. "
                "Il n'est **pas encore optimal** : les projections peuvent être imprécises, "
                "notamment pour les articles avec peu d'historique (moins de 30 semaines de données). "
                "Trois scénarios sont estimés pour chaque article : "
                "optimiste (q5), médian (q50) et pessimiste (q95). "
                "Plus l'historique est long, plus les prévisions seront fiables."
            )

        stocks_args: list[str] = ["--weeks", str(int(weeks))]
        stocks_env: dict[str, str] = {}
        if no_mail:
            stocks_args.append("--no-mail")
        if use_ml:
            stocks_args.append("--ml")
        try:
            safe_mock = _sanitize_mock_file(mock_file)
            if safe_mock:
                stocks_env["SUMUP_MOCK_FILE"] = safe_mock
        except ValueError as exc:
            st.error(str(exc))

        email_input = st.text_input(
            "Destinataires (séparés par des virgules)",
            value=_default_recipients(email_env_var),
            key=f"emails_{sid}",
            disabled=is_running,
            help="Valeurs par défaut issues de secrets.toml. Modifiez avant de lancer si besoin.",
        )

        if st.button("Lancer le rapport Stocks", key=f"btn_{sid}", disabled=is_running):
            cmd = build_cmd(cfg, stocks_args)
            run_script(sid, cmd, mail_env_var=email_env_var, email_override=email_input,
                       extra_env=stocks_env)

        _show_result(sid, skip_email=st.session_state.get(f"no_mail_{sid}", False), success_label="Rapport Stocks")

# ── Adhésions ─────────────────────────────────────────────────────────────────

elif sid == "adhesions":
    today = date.today()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=today - timedelta(days=14),
            key=f"start_{sid}",
            disabled=is_running,
            help="Premier jour de la période à analyser (inclus).",
        )
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=today,
            key=f"end_{sid}",
            disabled=is_running,
            help="Dernier jour de la période à analyser (inclus).",
        )

    filtres = st.text_input(
        "Mots-clés de filtre",
        value="adhesion",
        key=f"filtres_{sid}",
        disabled=is_running,
        placeholder="ex : Adhesion Don",
        help=(
            "Mots-clés recherchés dans le libellé des transactions SumUp, séparés par des espaces. "
            "Laisser vide pour inclure toutes les transactions sans filtre."
        ),
    )

    no_mail = st.checkbox(
        "Ne pas envoyer l'email (générer le PDF uniquement)",
        value=False,
        key=f"no_mail_{sid}",
        disabled=is_running,
    )

    adhesion_args: list[str] = ["--start", str(start_date), "--end", str(end_date)]
    adhesion_env: dict[str, str] = {}
    if no_mail:
        adhesion_args.append("--no-mail")
    try:
        tokens = _sanitize_filter_tokens(filtres)
        if tokens:
            adhesion_env["SUMUP_FILTRES"] = " ".join(tokens)
        elif filtres.strip() == "" and filtres != "":
            adhesion_env["SUMUP_FILTRES"] = ""
    except ValueError as exc:
        st.error(str(exc))

    email_input = st.text_input(
        "Destinataires (séparés par des virgules)",
        value=_default_recipients(email_env_var),
        key=f"emails_{sid}",
        disabled=is_running,
        help="Valeurs par défaut issues de secrets.toml.",
    )

    if st.button("Lancer le rapport Adhésions", key=f"btn_{sid}", disabled=is_running):
        cmd = build_cmd(cfg, adhesion_args)
        run_script(sid, cmd, mail_env_var=email_env_var, email_override=email_input,
                   extra_env=adhesion_env)

    _show_result(sid, skip_email=st.session_state.get(f"no_mail_{sid}", False), success_label="Rapport Adhésions")

# ── Membres (Paheko) ──────────────────────────────────────────────────────────

elif sid == "paheko":
    no_mail = st.checkbox(
        "Ne pas envoyer l'email (générer le tableau de bord uniquement)",
        value=False,
        key=f"no_mail_{sid}",
        disabled=is_running,
    )

    paheko_args: list[str] = []
    if no_mail:
        paheko_args.append("--no-mail")

    email_input = st.text_input(
        "Destinataires (séparés par des virgules)",
        value=_default_recipients(email_env_var),
        key=f"emails_{sid}",
        disabled=is_running,
        help="Valeurs par défaut issues de secrets.toml.",
    )

    if st.button("Lancer le tableau de bord Membres", key=f"btn_{sid}", disabled=is_running):
        cmd = build_cmd(cfg, paheko_args)
        run_script(sid, cmd, mail_env_var=email_env_var, email_override=email_input)

    _show_result(sid, skip_email=st.session_state.get(f"no_mail_{sid}", False), success_label="Tableau de bord")

# ── Statistiques ──────────────────────────────────────────────────────────────

elif sid == "stats":
    col1, col2 = st.columns([1, 2])
    with col1:
        weeks = st.number_input(
            "Semaines d'historique",
            min_value=1, max_value=52, value=8,
            key=f"weeks_{sid}",
            disabled=is_running,
            help="Nombre de semaines de transactions à analyser.",
        )
    with col2:
        mock_file = st.text_input(
            "Fichier mock (optionnel)",
            value="",
            key=f"mock_{sid}",
            disabled=is_running,
            placeholder="ex : stocks/mock_transactions.json",
            help="Chemin relatif vers un fichier JSON de transactions fictives.",
        )

    col3, col4 = st.columns(2)
    with col3:
        no_mail = st.checkbox(
            "Ne pas envoyer l'email",
            value=False,
            key=f"no_mail_{sid}",
            disabled=is_running,
        )
    with col4:
        no_enrich = st.checkbox(
            "Désactiver l'enrichissement",
            value=False,
            key=f"no_enrich_{sid}",
            disabled=is_running,
            help=(
                "Désactive les appels API supplémentaires par transaction "
                "(plus rapide, mais moins de détails dans le rapport)."
            ),
        )

    stats_args: list[str] = ["--weeks", str(int(weeks))]
    stats_env: dict[str, str] = {}
    if no_mail:
        stats_args.append("--no-mail")
    if no_enrich:
        stats_args.append("--no-enrich")
    try:
        safe_mock = _sanitize_mock_file(mock_file)
        if safe_mock:
            stats_env["SUMUP_MOCK_FILE"] = safe_mock
    except ValueError as exc:
        st.error(str(exc))

    email_input = st.text_input(
        "Destinataires (séparés par des virgules)",
        value=_default_recipients(email_env_var),
        key=f"emails_{sid}",
        disabled=is_running,
        help="Valeurs par défaut issues de secrets.toml.",
    )

    if st.button("Lancer les Statistiques", key=f"btn_{sid}", disabled=is_running):
        cmd = build_cmd(cfg, stats_args)
        run_script(sid, cmd, mail_env_var=email_env_var, email_override=email_input,
                   extra_env=stats_env)

    _show_result(sid, skip_email=st.session_state.get(f"no_mail_{sid}", False), success_label="Rapport Statistiques")
