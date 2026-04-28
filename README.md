# SumUp Auto Report

Outil de gestion des stocks et des rapports pour café associatif basé sur les transactions SumUp. Génère des rapports PDF hebdomadaires, suit les stocks, intègre les achats depuis Google Drive, et publie un tableau de bord Paheko.

## Prérequis

- Python 3.10+
- Git

## Installation

```bash
git clone https://github.com/SKOHscripts/sumup_auto_report.git
cd sumup_auto_report
pip install -r requirements.txt
```

Pour une installation en mode développement (avec les outils de lint et test) :

```bash
pip install -e ".[dev]"
```

## Configuration

### 1. Secrets et variables d'environnement

Copiez le fichier d'exemple et remplissez-le :

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Éditez `.streamlit/secrets.toml` avec vos valeurs :

| Variable | Description | Obligatoire |
|---|---|---|
| `APP_PASSWORD` | Mot de passe de l'interface web | Oui |
| `SUMUP_TOKEN` | Token API SumUp (`sup_sk_…`) | Oui |
| `EMAIL_ADDRESS` | Expéditeur SMTP (compte Gmail) | Pour l'email |
| `EMAIL_PASSWORD` | Mot de passe applicatif Google (16 car.) | Pour l'email |
| `EMAIL_TO_SUMUP_ALL_CA` | Destinataires rapports stocks + Paheko | Pour l'email |
| `EMAIL_TO_SUMUP_FINANCE` | Destinataires rapport adhésions | Pour l'email |
| `PAHEKO_BASE_URL` | URL de votre instance Paheko | Pour Paheko |
| `PAHEKO_API_USER` | Utilisateur API Paheko | Pour Paheko |
| `PAHEKO_API_PASSWORD` | Mot de passe API Paheko | Pour Paheko |
| `GDRIVE_SERVICE_ACCOUNT_FILE` | Chemin vers le JSON service account Google | Pour les achats |
| `GDRIVE_PURCHASES_FILE_ID` | ID du fichier Excel dans Google Drive | Pour les achats |

Pour les scripts CLI (cron), créez un fichier `.env` à la racine avec les mêmes variables :

```bash
cp .streamlit/secrets.toml.example .env
# Éditez .env — même format clé=valeur
```

### 2. Configuration Google Drive (module achats)

Pour intégrer automatiquement le fichier `ACHATS_suivi_stock.xlsx` :

1. Dans [Google Cloud Console](https://console.cloud.google.com/) :
   - Activer l'API **Google Drive**
   - Créer un **Service Account** (IAM & Admin → Comptes de service)
   - Télécharger la clé JSON du service account

2. Placer la clé dans le projet (ne pas committer) :

   ```bash
   mkdir -p .secrets
   cp /chemin/vers/ma-cle.json .secrets/gdrive_service_account.json
   ```

3. Dans Google Drive : partager le fichier Excel avec l'email du service account (`xxx@projet.iam.gserviceaccount.com`) en lecture seule.

4. Récupérer l'ID du fichier depuis son URL Drive :
   `https://docs.google.com/spreadsheets/d/**FILE_ID**/edit`

5. Renseigner dans `.streamlit/secrets.toml` (ou `.env`) :
   ```
   GDRIVE_SERVICE_ACCOUNT_FILE = "/chemin/absolu/.secrets/gdrive_service_account.json"
   GDRIVE_PURCHASES_FILE_ID    = "FILE_ID"
   ```

## Utilisation

### Interface web (Streamlit)

```bash
streamlit run app.py
```

Ouvre un dashboard sur `http://localhost:8501` permettant de lancer tous les modules avec options (dry-run, fichier mock, destinataires…).

### CLI — scripts disponibles

```bash
# Rapport stocks hebdomadaire
python -m stocks.sumup_stocks
python -m stocks.sumup_stocks --weeks 4 --no-mail

# Rapport adhésions
python -m adhesions.sumup_adhesions --start 2026-01-01 --end 2026-03-31

# Statistiques SumUp
python -m stocks.sumup_statistics --weeks 8

# Tableau de bord Paheko
python -m paheko_stats.paheko

# Intégration des achats depuis Google Drive
python -m stocks.update_stock_from_purchases
python -m stocks.update_stock_from_purchases --dry-run        # simulation
python -m stocks.update_stock_from_purchases --local FICHIER.xlsx  # fichier local
```

### Wrapper bash (avec git pull/push automatique)

```bash
chmod +x run.sh

./run.sh stocks              # rapport stocks + commit/push stock_items.json
./run.sh adhesions           # rapport adhésions (garde-fou 7 jours)
./run.sh paheko              # tableau de bord Paheko
./run.sh purchases           # intégration achats depuis Google Drive
./run.sh purchases --dry-run # simulation sans modification
```

### Automatisation via crontab

```cron
# Rapport stocks — chaque lundi à 09h00
0 9 * * 1  /home/user/sumup_auto_report/run.sh stocks

# Rapport adhésions — chaque jour à 08h00 (exécuté au plus une fois par semaine)
0 8 * * *  /home/user/sumup_auto_report/run.sh adhesions

# Tableau de bord Paheko — 1er du mois à 07h00
0 7 1 * *  /home/user/sumup_auto_report/run.sh paheko

# Intégration des achats — chaque jour à 10h00
0 10 * * * /home/user/sumup_auto_report/run.sh purchases
```

## Structure du projet

```
sumup_auto_report/
├── app.py                          # Interface Streamlit
├── run.sh                          # Wrapper bash (cron + git)
├── requirements.txt                # Dépendances Python
├── pyproject.toml                  # Métadonnées du projet
├── stocks/
│   ├── sumup_stocks.py             # Rapport stocks hebdomadaire
│   ├── sumup_statistics.py         # Statistiques de ventes
│   ├── update_stock_from_purchases.py  # Intégration achats Google Drive
│   ├── gdrive_loader.py            # Téléchargement Google Drive
│   ├── stock_items.json            # Catalogue + état des stocks
│   └── purchase_mapping.json       # Mapping produits Excel → stock_sku
├── adhesions/
│   └── sumup_adhesions.py          # Rapport adhésions
├── paheko_stats/
│   └── paheko.py                   # Tableau de bord membres
├── utils/
│   ├── mail_utils.py               # Email SMTP + chargement .env
│   └── sumup_shared.py             # Fonctions partagées
└── .streamlit/
    └── secrets.toml.example        # Modèle de configuration
```

## Module achats — fonctionnement

Le fichier `ACHATS_suivi_stock.xlsx` sur Google Drive est un tableau collaboratif où chaque colonne représente un achat (date + acheteur + quantités). Le module :

1. Télécharge le fichier depuis Google Drive via l'API v3
2. Parse les colonnes d'achat (les colonnes marquées `exemple` sont ignorées)
3. Fait correspondre les noms de produits Excel aux `stock_sku` via `stocks/purchase_mapping.json`
4. Ajoute les quantités achetées au `stock_on_hand` de `stock_items.json`
5. Trace chaque achat dans `stock_history` avec `type: "purchase"` (déduplication par date)

Pour ajouter un nouveau produit au fichier Excel, ajoutez une entrée dans `stocks/purchase_mapping.json` :

```json
{
  "excel_label": "Nom exact dans la colonne B de l'Excel",
  "stock_sku": "sku_dans_stock_items",
  "qty_multiplier": 1
}
```

## Dépendances

| Paquet | Usage |
|---|---|
| `streamlit` | Interface web |
| `requests` | API SumUp et Paheko |
| `fpdf2` | Génération PDF |
| `matplotlib` + `numpy` | Graphiques |
| `python-dateutil` | Parsing de dates |
| `python-dotenv` | Chargement `.env` |
| `openpyxl` | Lecture fichiers Excel |
| `google-api-python-client` | API Google Drive |
| `google-auth` | Authentification Google |
