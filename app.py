import os
import sys
import subprocess
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SumUp Reports", layout="centered")

# ── authentication ────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("SumUp Reports")
    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()

# ── configuration des scripts ─────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# email_env_var : variable d'environnement qui contrôle la liste de destinataires
SCRIPTS = {
    "stocks": {
        "title": "Rapport Stocks",
        "caption": "Inventaire hebdomadaire et alertes de réapprovisionnement",
        "description": (
            "Ce script récupère les transactions SumUp des dernières semaines, "
            "déduit les quantités vendues de chaque article, et génère un rapport PDF "
            "avec l'état des stocks, les seuils de réapprovisionnement et les alertes. "
            "**Il met aussi à jour le fichier `stock_items.json` et pousse les changements "
            "sur le dépôt git automatiquement.**"
        ),
        "use_run_sh": True,
        "email_env_var": "EMAIL_TO_SUMUP_ALL_CA",
    },
    "adhesions": {
        "title": "Rapport Adhésions",
        "caption": "Transactions d'adhésion et dons SumUp",
        "description": (
            "Extrait les transactions SumUp contenant des mots-clés définis (ex: « adhésion », « don ») "
            "sur une période donnée, et génère un PDF récapitulatif groupé par moyen de paiement "
            "(espèces, carte Visa, Mastercard…) avec totaux par section."
        ),
        "path": "adhesions/sumup_adhesions.py",
        "email_env_var": "EMAIL_TO_SUMUP_FINANCE",
    },
    "paheko": {
        "title": "Tableau de bord Paheko",
        "caption": "Statistiques et démographie des membres",
        "description": (
            "Se connecte à l'API Paheko pour récupérer les données des membres actifs "
            "et génère un tableau de bord visuel : répartition par catégorie, pyramide des âges, "
            "abonnements newsletter, villes représentées, et évolution des inscriptions dans le temps."
        ),
        "path": "paheko_stats/paheko.py",
        "email_env_var": "EMAIL_TO_SUMUP_ALL_CA",
    },
    "stats": {
        "title": "Statistiques SumUp",
        "caption": "Analyse des ventes et performance produits",
        "description": (
            "Analyse les transactions SumUp sur une période configurable pour produire un rapport "
            "sur le chiffre d'affaires par produit et par catégorie, la répartition des moyens de paiement, "
            "et l'évolution hebdomadaire des ventes. Identifie aussi les produits sans correspondance "
            "dans le catalogue (anomalies de données)."
        ),
        "path": "stocks/sumup_statistics.py",
        "email_env_var": "EMAIL_TO",
    },
}

for sid in SCRIPTS:
    st.session_state.setdefault(f"logs_{sid}", [])
    st.session_state.setdefault(f"running_{sid}", False)
    st.session_state.setdefault(f"rc_{sid}", None)


def build_env(email_overrides=None):
    env = dict(os.environ)

    # Passe tous les secrets au sous-processus (top-level + sections TOML imbriquées)
    for key, val in st.secrets.items():
        if isinstance(val, (str, int, float)):
            env[key] = str(val)
        elif hasattr(val, "items"):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, (str, int, float)):
                    env[sub_key] = str(sub_val)

    # Alias : noms dans secrets.toml → noms attendus par les scripts
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


def _default_recipients(env_var):
    """Retourne la liste de destinataires par défaut depuis secrets.toml."""
    return st.secrets.get(env_var, "")


def build_cmd(cfg, extra_args):
    """Construit la commande à exécuter selon le type de script."""
    if cfg.get("use_run_sh"):
        return ["./run.sh", "stocks"] + extra_args
    module = Path(cfg["path"]).with_suffix("").as_posix().replace("/", ".")
    return [sys.executable, "-m", module] + extra_args


def run_script(sid, cmd, email_env_var=None, email_override=None):
    st.session_state[f"logs_{sid}"] = []
    st.session_state[f"rc_{sid}"] = None
    st.session_state[f"running_{sid}"] = True

    overrides = {}
    if email_env_var and email_override and email_override.strip():
        overrides[email_env_var] = email_override.strip()

    env = build_env(email_overrides=overrides)
    log_area = st.empty()
    log_lines = []

    with st.spinner("Script en cours..."):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=BASE_DIR,
        )
        for line in process.stdout:
            log_lines.append(line)
            log_area.code("".join(log_lines))
        process.wait()

    log_area.code("".join(log_lines))
    st.session_state[f"logs_{sid}"] = log_lines
    st.session_state[f"rc_{sid}"] = process.returncode
    st.session_state[f"running_{sid}"] = False


# ── main UI ───────────────────────────────────────────────────────────────────

st.title("SumUp Reports")
st.caption(
    "Lancez les rapports manuellement. Chaque script génère un PDF (ou PNG) "
    "et l'envoie par email aux destinataires configurés, sauf si l'option « sans email » est cochée."
)

for i, (sid, cfg) in enumerate(SCRIPTS.items()):
    st.subheader(cfg["title"])
    st.caption(cfg["caption"])

    with st.expander("À propos de ce rapport", expanded=False):
        st.markdown(cfg["description"])

    is_running = st.session_state[f"running_{sid}"]
    email_env_var = cfg.get("email_env_var")
    extra_args = []

    # ── options spécifiques à chaque script ───────────────────────────────────

    if sid == "stocks":
        col1, col2 = st.columns([1, 2])
        with col1:
            weeks = st.number_input(
                "Semaines d'historique",
                min_value=1, max_value=52, value=4,
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
                    "Chemin vers un fichier JSON local de transactions. "
                    "Si renseigné, l'API SumUp n'est pas appelée — utile pour les tests."
                ),
            )
        no_mail = st.checkbox(
            "Ne pas envoyer l'email (générer le PDF uniquement)",
            value=False,
            key=f"no_mail_{sid}",
            disabled=is_running,
        )
        extra_args += ["--weeks", str(int(weeks))]
        if no_mail:
            extra_args.append("--no-mail")
        if mock_file.strip():
            extra_args += ["--mock", mock_file.strip()]

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
            "Mots-clés de filtre (optionnel)",
            value="",
            key=f"filtres_{sid}",
            disabled=is_running,
            placeholder="ex : Adhesion Don",
            help=(
                "Mots-clés recherchés dans le libellé des transactions, séparés par des espaces. "
                "Laisser vide = filtres par défaut du script (ex : « adhesion »). "
                "Entrer un espace = inclure toutes les transactions sans filtre."
            ),
        )
        no_mail = st.checkbox(
            "Ne pas envoyer l'email (générer le PDF uniquement)",
            value=False,
            key=f"no_mail_{sid}",
            disabled=is_running,
        )
        extra_args += ["--start", str(start_date), "--end", str(end_date)]
        if no_mail:
            extra_args.append("--no-mail")
        tokens = filtres.split()
        if tokens:
            extra_args += ["--filtres"] + tokens
        elif filtres.strip() == "" and filtres != "":
            # espace seul → --filtres sans valeur (toutes transactions)
            extra_args.append("--filtres")

    elif sid == "paheko":
        no_mail = st.checkbox(
            "Ne pas envoyer l'email (générer le dashboard uniquement)",
            value=False,
            key=f"no_mail_{sid}",
            disabled=is_running,
        )
        if no_mail:
            extra_args.append("--no-mail")

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
                help=(
                    "Chemin vers un fichier JSON local de transactions. "
                    "Si renseigné, l'API SumUp n'est pas appelée."
                ),
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
        extra_args += ["--weeks", str(int(weeks))]
        if no_mail:
            extra_args.append("--no-mail")
        if no_enrich:
            extra_args.append("--no-enrich")
        if mock_file.strip():
            extra_args += ["--mock", mock_file.strip()]

    # ── destinataires + bouton de lancement ───────────────────────────────────

    email_input = st.text_input(
        "Destinataires (séparés par des virgules)",
        value=_default_recipients(email_env_var) if email_env_var else "",
        key=f"emails_{sid}",
        disabled=is_running,
        help="Valeurs par défaut issues de secrets.toml. Modifiez avant de lancer si besoin.",
    )

    if st.button("Lancer", key=f"btn_{sid}", disabled=is_running):
        cmd = build_cmd(cfg, extra_args)
        run_script(sid, cmd, email_env_var=email_env_var, email_override=email_input)

    logs = st.session_state[f"logs_{sid}"]
    rc = st.session_state[f"rc_{sid}"]

    if logs:
        st.code("".join(logs))

    if rc is not None:
        if rc == 0:
            if st.session_state.get(f"no_mail_{sid}", False):
                st.success("Rapport généré avec succès (sans envoi email).")
            else:
                st.success("Rapport généré et email envoyé.")
        else:
            st.error("Erreur — voir les logs ci-dessus.")

    if i < len(SCRIPTS) - 1:
        st.divider()
