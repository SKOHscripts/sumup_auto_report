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
python -m stocks.sumup_stocks --ml                # avec projection ML quantile (P10/P50/P90)

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

# Pipeline Machine Learning (cf. docs/ML.md)
python -m stocks.ml.bootstrap --since 2025-12-01  # amorce le parquet (1 fois)
python -m stocks.ml.train --diagnose              # rapport par SKU
python -m stocks.ml.train --tune                  # tuning des hyperparamètres
python -m stocks.ml.train                         # entraînement + promotion hebdo
python -m stocks.ml.train --report                # journal des promotions
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
│   ├── sumup_stocks.py             # Rapport stocks hebdomadaire (flag --ml)
│   ├── sumup_statistics.py         # Statistiques de ventes
│   ├── update_stock_from_purchases.py  # Intégration achats Google Drive
│   ├── gdrive_loader.py            # Téléchargement Google Drive
│   ├── stock_items.json            # Catalogue + état des stocks
│   ├── purchase_mapping.json       # Mapping produits Excel → stock_sku
│   ├── ml/                         # Sous-système Machine Learning (cf. docs/ML.md)
│   │   ├── dataset.py              # Persistance hebdo (parquet)
│   │   ├── features.py             # Feature engineering (calendrier, lags, rolling)
│   │   ├── model.py                # Modèle quantile HGB + baseline Ridge
│   │   ├── projection.py           # Forecast itératif + Monte-Carlo de rupture
│   │   ├── evaluation.py           # Walk-forward backtest + métriques
│   │   ├── registry.py             # Archive + journal + détection de dérive
│   │   ├── inference.py            # Orchestrateur (sumup_stocks --ml)
│   │   ├── config.py               # MLConfig persistante
│   │   ├── tuning.py               # RandomizedSearchCV des hyperparamètres
│   │   ├── diagnose.py             # Rapport par SKU
│   │   ├── bootstrap.py            # CLI d'amorçage initial
│   │   └── train.py                # CLI hebdomadaire (train/tune/diagnose/report)
│   ├── data/                       # Historique persistant (généré)
│   │   └── weekly_usage.parquet
│   └── models/                     # Modèles versionnés (généré)
│       ├── current.joblib
│       ├── archive/<YYYY-WNN>/
│       ├── config.json
│       └── history.csv
├── adhesions/
│   └── sumup_adhesions.py          # Rapport adhésions
├── paheko_stats/
│   └── paheko.py                   # Tableau de bord membres
├── utils/
│   ├── mail_utils.py               # Email SMTP + chargement .env
│   └── sumup_shared.py             # Fonctions partagées
├── docs/
│   └── ML.md                       # Documentation détaillée du sous-système ML
└── .streamlit/
    └── secrets.toml.example        # Modèle de configuration
```

## Module Machine Learning — vue rapide

Le rapport stocks peut s'enrichir d'une projection probabiliste de la
date de rupture (P10 / P50 / P90) avec bande de confiance dans le PDF.
Le sous-système est **opt-in** via `--ml` et **best-effort** : si
l'historique est trop court ou les dépendances manquent, le rapport
retombe sur le calcul historique.

### Pipeline en 3 commandes

```bash
# 1. Une seule fois — amorce le parquet d'historique
python -m stocks.ml.bootstrap --since 2025-12-01

# 2. Chaque semaine — entraîne, évalue, promeut, alerte si dérive
python -m stocks.ml.train

# 3. Génère le rapport avec la projection ML
python -m stocks.sumup_stocks --ml
```

### Fonctionnalités clés

- **Persistance hebdo idempotente** des consommations dans
  `stocks/data/weekly_usage.parquet`.
- **Feature engineering** : calendrier (jours fériés FR), lags
  (t-1, t-2, t-4, t-12), moyennes/écarts mobiles, encodage cyclique.
  Anti-leakage strict.
- **Modèle quantile multi-SKU** (`HistGradientBoostingRegressor` ×3)
  avec `stock_sku` en feature catégorielle native.
- **Monte-Carlo** : 1 000 trajectoires de stock pour estimer la
  distribution de la date de rupture.
- **Walk-forward backtest** + règle de promotion (MAPE < seuil,
  coverage dans cible±tolérance, MAPE < baseline).
- **Versioning** : un modèle archivé par semaine ISO sous
  `stocks/models/archive/<YYYY-WNN>/`, symlink `current.joblib`.
- **Journal CSV** de toutes les évaluations + détection de dérive
  (3 semaines consécutives au-dessus du seuil).
- **Tuning** d'hyperparamètres via `RandomizedSearchCV` +
  `TimeSeriesSplit`, persistance dans `models/config.json`.
- **Diagnostic par SKU** pour identifier les SKU qui plombent les
  métriques (n_zéros, CV, MAPE individuelle).

### Sous-commandes utiles

```bash
python -m stocks.ml.train --diagnose          # rapport par SKU trié par MAPE
python -m stocks.ml.train --tune              # recherche d'hyperparamètres
python -m stocks.ml.train --report            # journal des 10 dernières évaluations
python -m stocks.ml.train --force             # promeut malgré les seuils (exploration)

# Configurables persistés dans models/config.json :
python -m stocks.ml.train --quantiles 0.05,0.5,0.95
python -m stocks.ml.train --mape-threshold 0.45 --coverage-tolerance 0.10
```

📖 **Documentation complète** : [`docs/ML.md`](docs/ML.md)
(architecture, métriques, dépannage, décisions techniques, tests).

## Module achats — fonctionnement

Le fichier `ACHATS_suivi_stock.xlsx` sur Google Drive est un tableau collaboratif où chaque colonne représente un achat (date + acheteur + quantités). Le module :

1. Télécharge le fichier depuis Google Drive via l'API v3
2. Parse les colonnes d'achat (les colonnes marquées `exemple` sont ignorées)
3. Fait correspondre les noms de produits Excel aux `stock_sku` via `stocks/purchase_mapping.json`
4. Ajoute les quantités achetées au `stock_on_hand` de `stock_items.json`
5. Trace chaque achat dans `stock_history` avec `type: "purchase"` (déduplication par date)

### Colonnes « état des lieux » (comptage physique)

Par défaut, chaque colonne est un **achat** : ses quantités sont **ajoutées** au stock.
Pour déclarer plutôt qu'un **comptage physique** (état des lieux) a été réalisé à une date,
il suffit d'écrire `état des lieux` (ou `inventaire`) dans la **ligne 2** de la colonne —
la même ligne que les marqueurs `exemple`, juste au-dessus du prénom et de la date.

Dans ce cas, le module :

- **Remplace** le `stock_on_hand` par la quantité comptée (au lieu de l'ajouter) ;
  une cellule à `0` vide donc effectivement le stock du produit (un produit laissé vide
  dans la colonne n'est pas modifié).
- Ré-ancre `last_inventory_date` et `last_auto_update` à la date du comptage, pour que le
  rapport de stocks ne déduise ensuite que les ventes **postérieures** à l'état des lieux.
- Trace l'opération dans `stock_history` avec `type: "inventory"` (déduplication par date).

Cela permet à n'importe qui de corriger les stocks directement depuis l'Excel, sans toucher au code.

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
| `pandas` + `pyarrow` | Persistance hebdo (parquet) — module ML |
| `scikit-learn` | HistGradientBoosting quantile, RandomizedSearchCV — module ML |
| `joblib` | Sérialisation des modèles — module ML |
