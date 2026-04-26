import os
import sys
import subprocess

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

# ── helpers ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# email_env_var : variable d'environnement qui contrôle la liste de destinataires
SCRIPTS = {
    "stocks": {
        "title": "Rapport Stocks",
        "caption": "Inventaire hebdomadaire et alertes de réapprovisionnement",
        "path": "stocks/sumup_stocks.py",
        "email_env_var": "EMAIL_TO_SUMUP_ALL_CA",
    },
    "adhesions": {
        "title": "Rapport Adhésions",
        "caption": "Transactions d'adhésion et dons SumUp",
        "path": "adhesions/sumup_adhesions.py",
        "email_env_var": "EMAIL_TO_SUMUP_FINANCE",
    },
    "paheko": {
        "title": "Tableau de bord Paheko",
        "caption": "Statistiques et démographie des membres",
        "path": "paheko_stats/paheko.py",
        "email_env_var": "EMAIL_TO_SUMUP_ALL_CA",
    },
    "stats": {
        "title": "Statistiques SumUp",
        "caption": "Analyse des ventes et performance produits",
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

    # Surcharge des destinataires saisie par l'utilisateur dans l'UI
    if email_overrides:
        env.update(email_overrides)

    return env


def _default_recipients(env_var):
    """Retourne la liste de destinataires par défaut depuis secrets.toml."""
    return st.secrets.get(env_var, "")


def run_script(sid, script_path, email_env_var=None, email_override=None):
    st.session_state[f"logs_{sid}"] = []
    st.session_state[f"rc_{sid}"] = None
    st.session_state[f"running_{sid}"] = True

    cmd = [sys.executable, os.path.join(BASE_DIR, script_path)]

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

for i, (sid, cfg) in enumerate(SCRIPTS.items()):
    st.subheader(cfg["title"])
    st.caption(cfg["caption"])

    is_running = st.session_state[f"running_{sid}"]
    email_env_var = cfg.get("email_env_var")

    email_input = st.text_input(
        "Destinataires (séparés par des virgules)",
        value=_default_recipients(email_env_var) if email_env_var else "",
        key=f"emails_{sid}",
        disabled=is_running,
        help="Valeurs par défaut issues de secrets.toml. Modifiez avant de lancer si besoin.",
    )

    if st.button("Lancer", key=f"btn_{sid}", disabled=is_running):
        run_script(sid, cfg["path"], email_env_var=email_env_var, email_override=email_input)

    logs = st.session_state[f"logs_{sid}"]
    rc = st.session_state[f"rc_{sid}"]

    if logs:
        st.code("".join(logs))

    if rc is not None:
        if rc == 0:
            st.success("Email envoyé.")
        else:
            st.error("Erreur — voir les logs ci-dessus.")

    if i < len(SCRIPTS) - 1:
        st.divider()
