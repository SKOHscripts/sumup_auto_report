# Machine Learning — projection probabiliste de la date de rupture

> Documentation détaillée du sous-système ML qui enrichit le rapport
> hebdomadaire des stocks avec une **distribution** de la date de rupture
> (P10 / P50 / P90) au lieu d'une simple date ponctuelle.

## Sommaire

1. [Pourquoi ce module](#pourquoi-ce-module)
2. [Vue d'ensemble](#vue-densemble)
3. [Concepts clés](#concepts-clés)
4. [Architecture](#architecture)
5. [Workflow opérationnel](#workflow-opérationnel)
6. [CLI — référence complète](#cli--référence-complète)
7. [Fichiers générés](#fichiers-générés)
8. [Configuration persistante](#configuration-persistante)
9. [Lecture des métriques & diagnostic](#lecture-des-métriques--diagnostic)
10. [Dépannage](#dépannage)
11. [Décisions techniques](#décisions-techniques)
12. [Tests](#tests)

---

## Pourquoi ce module

### Le problème

Le calcul historique de la date de rupture, dans `stocks/sumup_stocks.py`,
est une simple division :

```python
coverage_weeks = effective_stock_now / avg_rolling4
rupture_date = now + coverage_weeks
```

Où `avg_rolling4` est la moyenne des 4 dernières semaines de consommation.
C'est une **estimation ponctuelle** sans saisonnalité ni intervalle de
confiance. Pour un café associatif où les ventes varient fortement entre
juillet et novembre, ou entre une semaine de vacances et une semaine
événementielle, cette approche perd beaucoup d'information.

### Ce que le ML apporte

- **Saisonnalité** : le modèle apprend les patterns de mois, semaine de
  l'année, jours fériés, position dans le mois.
- **Mémoire long terme** : lags hebdomadaires (t-1, t-2, t-4, t-12) et
  moyennes mobiles (4 et 13 semaines).
- **Incertitude** : trois quantiles (low / med / high) entraînés
  séparément donnent une fourchette, pas un point.
- **Date de rupture probabiliste** : 1 000 trajectoires Monte-Carlo
  donnent les P10 / P50 / P90 de la date de rupture.
- **Boucle d'amélioration continue** : auto-évaluation par walk-forward,
  archivage versionné, détection de dérive, journal CSV.

Le système est **best-effort** : si l'historique est trop court, si
pyarrow/scikit-learn manquent, si le modèle ne bat pas la baseline, le
rapport retombe sur le calcul historique sans casser.

---

## Vue d'ensemble

```
                ┌───────────────────────────────────────────┐
                │  API SumUp + stock_items.json (existant)  │
                └────────────────────┬──────────────────────┘
                                     │
            ┌────────────────────────▼──────────────────────────┐
            │  agrégation hebdo (semaine ISO × SKU)              │
            └──────────┬────────────────────────────┬────────────┘
                       │                            │
                       ▼                            ▼
        stocks/data/weekly_usage.parquet     KPIs (rapport actuel)
            (historique persistant)
                       │
                       ▼
       ┌─────────────────────────────┐
       │  features (calendrier+lags) │
       └─────────────┬───────────────┘
                     │
                     ▼
       ┌─────────────────────────────┐
       │  HGB quantile (low/med/high)│ ←── tuning halving (successive halving)
       └─────┬─────────────────┬─────┘     (–-tune, persisté)
             │                 │
             ▼                 ▼
   walk-forward backtest    forecast itératif N-step
   MAPE / coverage          (q_low/q_med/q_high par semaine)
             │                 │
             ▼                 ▼
   promote_if_better     simulate_rupture (Monte-Carlo 1000)
   (config.mape_threshold)    │
             │                 ▼
             │         rupture_date_p10 / p50 / p90
             ▼                 │
   models/archive/<sem>/       │
   models/current.joblib ←─────┘
   models/history.csv
                     │
                     ▼
       PDF avec bande de confiance + intervalle dans le tableau KPI
```

---

## Concepts clés

### Quantile regression

Au lieu d'entraîner un modèle de régression classique qui prédit
`E[y | X]`, on entraîne **trois modèles** qui prédisent chacun un
quantile conditionnel : `q5(y | X)`, `q50(y | X)` et `q95(y | X)`.

`HistGradientBoostingRegressor` de scikit-learn supporte nativement
`loss="quantile"` avec un paramètre `quantile=0.05`, ce qui revient à
optimiser la **pinball loss** :

```
L_q(y, ŷ) = max(q · (y − ŷ), (q − 1) · (y − ŷ))
```

Asymétrique : pour q=0.05, sur-prédire est 19× plus pénalisé que
sous-prédire ; le modèle apprend donc à produire un quantile bas.

### Walk-forward backtest

Pour évaluer honnêtement un modèle de série temporelle, on **interdit**
au modèle de voir le futur :

```
fold 1 : train sur t∈[0..50],  test sur t∈[50..56]
fold 2 : train sur t∈[0..56],  test sur t∈[56..62]
fold 3 : train sur t∈[0..62],  test sur t∈[62..68]
...
```

À chaque fold on entraîne un modèle frais sur le passé et on mesure
MAPE, MAE, pinball loss, **coverage** (% d'observations tombant dans
[q_low, q_high]). On agrège ensuite les métriques.

### Monte-Carlo de la date de rupture

À partir de la prévision quantile par semaine, on simule 1 000
trajectoires de stock :

```
pour chaque trajectoire :
    stock = stock_initial
    pour chaque semaine future t :
        si t = semaine_arrivée_commande:
            stock += incoming_qty
        consommation_t = sample(q_low_t, q_med_t, q_high_t)  # interpolation linéaire
        stock −= consommation_t
        si stock ≤ 0 et pas encore en rupture:
            enregistrer la semaine de rupture
```

On calcule ensuite les percentiles P10/P50/P90 sur les 1 000 semaines
de rupture observées. P50 = scénario médian, P10 = scénario pessimiste,
P90 = scénario optimiste.

### Détection de dérive

Toutes les évaluations (promues ou non) sont journalisées dans
`models/history.csv`. Si la MAPE dépasse le seuil pendant 3 semaines
consécutives, `detect_drift()` lève une alerte (code de sortie 3 du
CLI `train`).

---

## Architecture

```
stocks/
├── data/
│   └── weekly_usage.parquet         # Historique persistant (créé au bootstrap)
├── models/
│   ├── current.joblib               # Symlink vers le modèle promu
│   ├── current.joblib.meta.json     # Symlink vers ses métadonnées
│   ├── archive/
│   │   ├── 2026_W18/
│   │   │   ├── model.joblib         # Modèle archivé de la semaine
│   │   │   └── model.joblib.meta.json
│   │   └── 2026_W19/...
│   ├── config.json                  # MLConfig persistée (quantiles, seuils, params tunés)
│   └── history.csv                  # Journal de toutes les évaluations
└── ml/
    ├── __init__.py
    ├── dataset.py        # Persistance parquet idempotente
    ├── features.py       # Calendrier + lags + rolling, anti-leakage
    ├── model.py          # RidgeForecaster + QuantileGradientBoostingForecaster
    ├── projection.py     # forecast_horizon (N-step) + simulate_rupture (MC)
    ├── evaluation.py     # walk-forward backtest, MAPE / pinball / coverage
    ├── registry.py       # archive / current / journal / drift detection
    ├── inference.py      # Orchestrateur appelé par sumup_stocks.py --ml
    ├── config.py         # MLConfig (load/save JSON)
    ├── tuning.py         # Halving (Halving[Random|Grid]SearchCV) TimeSeriesSplit
    ├── diagnose.py       # Rapport par SKU
    ├── bootstrap.py      # CLI d'amorçage initial depuis l'API SumUp
    └── train.py          # CLI hebdomadaire train + tune + diagnose + report
```

### Responsabilités des modules

| Module | Responsabilité | Sortie |
|---|---|---|
| `dataset.py` | Persistance hebdo idempotente (clé `(stock_sku, week_label)`) | `weekly_usage.parquet` |
| `features.py` | Feature engineering anti-leakage | `(X, y, meta)` pour entraînement |
| `model.py` | Modèles ML (`RidgeForecaster` baseline + `QuantileGBM` principal) | Modèles entraînables/sérialisables |
| `projection.py` | Forecast N-step itératif + Monte-Carlo de la rupture | DataFrame + dict P10/P50/P90 |
| `evaluation.py` | Walk-forward backtest, métriques, règle de promotion | `EvaluationMetrics` |
| `registry.py` | Versioning des modèles + journal de promotion | `archive/<sem>/`, `current.joblib`, `history.csv` |
| `inference.py` | Orchestrateur : charge modèle, projette par SKU, attache aux KPIs | KPIs enrichis avec `ml_projection` |
| `config.py` | Configuration persistante (`MLConfig`) | `config.json` |
| `tuning.py` | Recherche d'hyperparamètres par halving (échantillon ou exhaustif) | Params persistés dans config, si meilleurs que l'actuel |
| `diagnose.py` | Rapport par SKU pour identifier les problèmes | DataFrame trié par MAPE |
| `bootstrap.py` | CLI d'amorçage initial depuis l'API SumUp | parquet rempli |
| `train.py` | CLI hebdomadaire principal | promotion + journal + alerte |

---

## Workflow opérationnel

### 1. Amorçage initial (une seule fois)

Récupère l'historique disponible côté API SumUp et remplit
`stocks/data/weekly_usage.parquet`.

```bash
python -m stocks.ml.bootstrap --since 2025-12-01
```

Options utiles :
- `--mock fichier.json` : utilise un dump local (tests)
- `--no-enrich` : saute l'enrichissement détail des transactions (rapide)
- `--dry-run` : affiche ce qui serait écrit sans toucher au parquet

### 2. Diagnostic préalable

Avant de tuner, regarder à quoi ressemblent les données :

```bash
python -m stocks.ml.train --diagnose
```

Sortie type :

```
SKU              n_sem  n_0  %_0   mean   std   CV     moy_4sem  MAPE_naive  MAPE_avg4
---------------  -----  ---  ----  -----  ----  -----  --------  ----------  ---------
sporadique          21   18   86%   0.71  1.79  2.52       0.50        220%       180%
volatil             21    2   10%   8.73  6.21  0.71      10.20         85%        72%
regulier            21    0    0%  10.05  0.55  0.05      10.10         12%         9%
```

Lecture :
- **CV > 1.5** ou **% zéros > 50 %** : SKU à demande intermittente,
  le ML ne saura pas faire mieux que la moyenne. Penser à exclure
  ou à fusionner avec un SKU parent.
- **MAPE_avg4** est le score à battre. Si même la baseline est à 70 %,
  les données sont intrinsèquement difficiles, pas le modèle.

Sauve aussi en CSV :

```bash
python -m stocks.ml.train --diagnose --diagnose-csv diag.csv
```

### 3. Tuning des hyperparamètres (occasionnel)

```bash
python -m stocks.ml.train --tune
```

#### Le halving, expliqué simplement

Tuner, c'est chercher le meilleur réglage (profondeur des arbres,
régularisation, etc.) parmi des centaines de combinaisons possibles.
Entraîner **un seul** modèle « à fond » (beaucoup d'arbres) est déjà
coûteux ; en tester des centaines à fond serait bien trop long.

Le tuning fonctionne comme les qualifications d'un tournoi à
élimination plutôt qu'un match complet pour chaque joueur :

| Tour | Candidats restants | Budget par candidat |
|---|---|---|
| 1 | 300 | petit (peu d'arbres) |
| 2 | 100 | moyen |
| 3 | 33 | plus grand |
| Finale | ~11 | plein régime |

À chaque tour, tous les candidats survivants sont entraînés avec un
**peu plus** de budget, puis seul le **meilleur tiers** passe au tour
suivant (le facteur d'élimination, `factor=3`, est configurable). Les
réglages manifestement mauvais sont éliminés tôt, à bas coût ; seuls
les prometteurs reçoivent le traitement complet. C'est l'algorithme
`HalvingRandomSearchCV` de scikit-learn, avec `max_iter` (le nombre
d'arbres) comme « ressource » qui augmente à chaque tour.

Résultat mesuré sur l'historique du projet : **~10× plus rapide**
qu'un `RandomizedSearchCV` classique à effort constant — donc, à temps
égal, on peut tester **bien plus** de combinaisons.

#### Déroulé complet d'un `--tune`

1. **Échantillonnage** : `--n-candidates 300` (défaut) tire 300
   combinaisons au hasard dans la grille et les passe au tournoi.
   `--exhaustive` balaye **toute** la grille (≈1000 combinaisons) au
   lieu d'un échantillon — plus complet, toujours accéléré par le
   halving. `--jobs -1` (défaut) répartit les entraînements sur tous
   les cœurs disponibles.
2. **Score sur la cible transformée** : le modèle s'entraîne (et se
   note) en espace `log1p(usage)` — cohérent avec le modèle réellement
   déployé (cf. « Modèle global multi-SKU plutôt qu'un par SKU » dans
   les décisions techniques plus bas), et équilibré entre SKU à faible
   et fort volume.
3. **Garde-fou anti-régression** : avant d'adopter le gagnant du
   tournoi, un vrai backtest walk-forward compare sa MAPE à celle de
   la config **actuelle**. Le nouveau jeu n'est adopté **que s'il fait
   réellement mieux** ; sinon la config est laissée inchangée. La CV
   interne du tuning ne capture pas toujours la généralisation — sans
   ce filet, `--tune` pourrait dégrader une config déjà bonne.
4. Si adopté, les paramètres sont **persistés** dans
   `stocks/models/config.json` et réutilisés automatiquement par
   toutes les exécutions futures (`train`, `inference`).

`--tune` peut donc être relancé sans risque : au pire, il ne change
rien. À relancer surtout après une grosse évolution de l'activité, ou
quand l'historique s'est significativement allongé.

### 4. Entraînement hebdomadaire

```bash
python -m stocks.ml.train
```

Pipeline :
1. Charge `weekly_usage.parquet` et la config.
2. Walk-forward backtest 5 plis avec les hyperparamètres tunés.
3. Calcule la MAPE de la baseline `avg_rolling4` pour comparaison.
4. Décide si le modèle est promotable (MAPE < seuil, coverage dans
   cible±tolérance, MAPE < baseline).
5. Entraîne le modèle final sur tout l'historique.
6. Archive sous `models/archive/<semaine>/`.
7. Si promotable → met à jour `models/current.joblib` (symlink).
8. Journalise dans `models/history.csv` (même si non promu).
9. Vérifie la dérive sur les 3 dernières semaines.

Codes de sortie :
- `0` : modèle promu (ou journal mis à jour avec `--no-promote`)
- `1` : pas d'historique persistant
- `2` : pas assez de données pour évaluer
- `3` : alerte de dérive
- `4` : non promu (qualité insuffisante)

### 5. Génération du rapport avec ML

```bash
python -m stocks.sumup_stocks --ml
```

Lit `models/current.joblib` si présent (sinon entraîne à la volée),
projette 26 semaines pour chaque SKU, simule la date de rupture, et
ajoute deux cellules au tableau KPI ainsi qu'une bande de confiance
violette sur le graphique d'évolution.

### 6. Consultation du journal

```bash
python -m stocks.ml.train --report
```

Affiche les 10 dernières évaluations :

```
Dernières évaluations :
date                 week       promu  MAPE     coverage   baseline
2026-05-12T07:00:01  2026-W19   OUI    0.35     0.82       0.50
2026-05-05T07:00:02  2026-W18   NON    0.81     0.45       0.71
...
```

---

## CLI — référence complète

### `stocks.ml.bootstrap`

```bash
python -m stocks.ml.bootstrap [OPTIONS]

  --since YYYY-MM-DD     Date de début (défaut : 2025-12-01)
  --items FICHIER        Chemin alternatif vers stock_items.json
  --mock FICHIER.json    Mode hors ligne, transactions depuis un dump JSON
  --no-enrich            Saute l'API d'enrichissement (test rapide)
  --dry-run              N'écrit pas le parquet, affiche un échantillon
  --output PATH          Chemin de sortie alternatif
```

### `stocks.ml.train`

```bash
python -m stocks.ml.train [SOUS-COMMANDE] [FLAGS]
```

**Sous-commandes** (mutuellement exclusives) :

| Flag | Effet |
|---|---|
| _(aucune)_ | Pipeline standard : backtest → train → promotion |
| `--tune` | Tuning par halving (cf. ci-dessus) puis pipeline standard |
| `--diagnose` | Rapport par SKU (n_sem, %_0, CV, MAPE) puis sort |
| `--report` | Affiche les 10 dernières lignes de `history.csv` puis sort |

**Modificateurs du train** :

| Flag | Effet |
|---|---|
| `--force` | Promeut même si critères non atteints (pour exploration) |
| `--no-promote` | Calcule les métriques mais n'archive rien |
| `--diagnose-csv FICHIER` | Avec `--diagnose` : sauve en CSV |

**Modificateurs du tuning** (avec `--tune`) :

| Flag | Effet |
|---|---|
| `--n-candidates 300` | Nombre de combinaisons échantillonnées pour le halving |
| `--exhaustive` | Balaye toute la grille (`HalvingGridSearchCV`) au lieu d'un échantillon |
| `--jobs -1` | Nombre de cœurs pour le tuning (`-1` = tous, défaut) |

**Configuration persistante** (s'écrit dans `models/config.json`) :

| Flag | Effet |
|---|---|
| `--quantiles q1,q2,q3` | Triplet `q_low,q_med,q_high` (ex : `0.05,0.5,0.95`) |
| `--mape-threshold 0.45` | Seuil MAPE max pour promotion (repli si pas de baseline) |
| `--coverage-target 0.80` | Couverture cible de l'intervalle |
| `--coverage-tolerance 0.15` | Tolérance autour de la cible |

### `stocks.sumup_stocks`

Pas de nouveau flag bloquant : seul `--ml` active l'enrichissement.

```bash
python -m stocks.sumup_stocks --ml
python -m stocks.sumup_stocks --ml --weeks 8 --no-mail
```

---

## Fichiers générés

### `stocks/data/weekly_usage.parquet`

Schéma :

| Colonne | Type | Exemple | Description |
|---|---|---|---|
| `stock_sku` | string | `chips` | Identifiant du SKU |
| `week_label` | string | `2026-W18` | Étiquette ISO |
| `year` | int32 | 2026 | Année ISO |
| `week` | int32 | 18 | Semaine ISO (1–53) |
| `week_start` | date | 2026-05-04 | Lundi de la semaine |
| `usage` | float64 | 12.5 | Quantité consommée (avec `consumption_per_sale`) |
| `sales_count` | int64 | 12 | Nombre de ventes brutes |

Clé unique : `(stock_sku, week_label)`. Les écritures sont
**idempotentes** : relancer `--ml` ou `bootstrap` sur la même semaine
remplace la ligne sans créer de doublon.

### `stocks/models/current.joblib`

Symlink vers le dernier modèle promu sous
`stocks/models/archive/<sem>/model.joblib`. Lisible par
`QuantileGradientBoostingForecaster.load(path)`.

### `stocks/models/<modèle>.meta.json`

```json
{
  "trained_at": "2026-05-12T07:00:00+00:00",
  "sklearn_version": "1.5.2",
  "n_samples": 420,
  "n_skus": 35,
  "n_features": 16,
  "config_hash": "ba281d329631",
  "metrics": {},
  "notes": ""
}
```

### `stocks/models/history.csv`

Une ligne par appel à `train` (promu ou non) :

| Colonne | Description |
|---|---|
| `promoted_at` | ISO timestamp |
| `week_label` | Semaine ISO |
| `version` | Hash de configuration du modèle (`config_hash`) |
| `promoted` | `1` si modèle promu, `0` sinon |
| `mae`, `mape` | Erreurs sur le quantile médian |
| `pinball_low/med/high` | Pinball loss par quantile |
| `coverage_band` | Proportion d'observations dans [q_low, q_high] |
| `baseline_mape` | MAPE de la baseline `avg_rolling4` |
| `n_samples`, `n_folds` | Tailles d'évaluation |
| `reasons` | Si non promu, raisons séparées par ` \| ` |

---

## Configuration persistante

Fichier `stocks/models/config.json`, géré par `stocks/ml/config.py` :

```json
{
  "quantiles": [0.05, 0.5, 0.95],
  "mape_threshold": 0.45,
  "coverage_target": 0.80,
  "coverage_tolerance": 0.15,
  "relative_mape_margin": 0.10,
  "target_transform": "log1p",
  "tuned_params": {
    "max_iter": 125,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_samples_leaf": 10,
    "l2_regularization": 10.0,
    "max_leaf_nodes": 7
  },
  "tuned_at": "2026-05-12T07:00:00+00:00",
  "tuning_score": 0.182
}
```

| Clé | Effet |
|---|---|
| `quantiles` | Triplet `(low, 0.5, high)`. Le médian doit valoir 0.5. |
| `mape_threshold` | Seuil MAPE max, utilisé seulement si aucune baseline n'est disponible |
| `coverage_target` ± `coverage_tolerance` | Plage acceptable de la couverture P_low–P_high |
| `relative_mape_margin` | Marge tolérée au-dessus de la baseline pour promouvoir (cf. règle de promotion) |
| `target_transform` | Transformation de la cible avant entraînement (`"log1p"` ou `null`) |
| `tuned_params` | Hyperparamètres HGB issus du dernier `--tune` **adopté** (le tuning ne persiste que s'il améliore le backtest) |
| `tuned_at` / `tuning_score` | Métadonnées du tuning |

Toutes les valeurs absentes sont remplacées par les défauts. Les flags
CLI écrivent dans ce fichier ; les exécutions suivantes les relisent
automatiquement.

---

## Lecture des métriques & diagnostic

### Métriques d'évaluation

| Métrique | Bonne valeur | Lecture |
|---|---|---|
| **MAPE** | < 30–40 % | Erreur relative médiane. > 60 % = données très bruitées |
| **MAE** | dépend du SKU | Erreur absolue. Plus pertinent pour SKU à faible volume |
| **Coverage band** | ≈ 80 % | % d'observations dans [q_low, q_high]. < 50 % = modèle sur-confiant ; > 95 % = modèle inutile car bande trop large |
| **Pinball loss** | minimiser | Score asymétrique propre aux modèles quantile |
| **baseline_mape** | référence | Si MAPE_ML ≥ baseline, le ML n'apporte rien |

### Règle de promotion

Un modèle est promu si **les deux** conditions sont réunies :

1. **Précision**, relative à la baseline si elle est disponible :
   `MAPE ≤ baseline_mape × (1 + relative_mape_margin)` (marge 0.10 par
   défaut). Sur une demande faible et erratique, un seuil MAPE absolu
   serait souvent inatteignable (la baseline elle-même peut dépasser
   60–70 %) ; le critère relatif reste pertinent en assumant une
   précision ponctuelle proche de la baseline, la valeur ajoutée du
   ML étant surtout des intervalles calibrés. Sans baseline, on
   retombe sur le seuil absolu `mape_threshold` (0.45 par défaut).
2. `|coverage − coverage_target| ≤ coverage_tolerance` (config,
   0.80 ± 0.15 par défaut).

### Workflow de mise au point

```bash
# 1. Comprendre les SKU difficiles
python -m stocks.ml.train --diagnose

# 2. Si certains SKU sont catastrophiques (CV > 2, %_0 > 70%) :
#    les exclure de stock_items.json ou les tagger comme non-prédictibles

# 3. Tuner les hyperparamètres
python -m stocks.ml.train --tune --n-candidates 300

# 4. Si la coverage reste basse, élargir l'intervalle
python -m stocks.ml.train --quantiles 0.05,0.5,0.95   # défaut
# ou plus large encore :
python -m stocks.ml.train --quantiles 0.025,0.5,0.975

# 5. Si l'historique est court, relâcher temporairement le seuil
python -m stocks.ml.train --mape-threshold 0.65 --coverage-tolerance 0.20

# 6. Si on accepte un modèle imparfait pour visualiser la bande
python -m stocks.ml.train --force
```

---

## Dépannage

### « Aucun historique persistant »

Le parquet n'existe pas. Lancer le bootstrap :

```bash
python -m stocks.ml.bootstrap --since 2025-12-01
```

### « Pas assez de donnees pour evaluer »

Le walk-forward exige au moins `min_train_size + n_folds = 55` lignes
post-warmup. Avec un historique court, attendre que `--ml` alimente le
parquet semaine après semaine.

### « MAPE trop élevé », « Coverage hors cible »

Cf. la section [Workflow de mise au point](#workflow-de-mise-au-point)
ci-dessus. C'est le comportement normal quand l'historique est court
ou très bruité.

### « ALERTE DRIFT »

Trois semaines consécutives de MAPE > seuil. Investiguer :

```bash
python -m stocks.ml.train --report
python -m stocks.ml.train --diagnose
```

Causes typiques : changement d'activité (nouveau menu, événement
spécial saisonnier), changement de la définition des SKU, problème
d'inventaire qui pollue les semaines récentes.

### Le rapport `--ml` ne montre pas la bande

Trois cas possibles :
1. Le SKU a moins de `MIN_WEEKS_PER_SKU = 16` semaines d'historique.
2. Aucun modèle promu : `models/current.joblib` n'existe pas. Lancer
   `train --force` ou attendre la prochaine promotion.
3. Une dépendance manque : pyarrow, scikit-learn. Voir `pip list`.

### CVE / mise à jour des dépendances

```bash
pip install --upgrade scikit-learn pyarrow pandas
```

Versions minimales actuelles :
- `pyarrow >= 14.0.1` (CVE-2023-47248)
- `scikit-learn >= 1.5.0` (CVE-2024-5206)

---

## Décisions techniques

### Modèle global multi-SKU plutôt qu'un par SKU

**Choix** : un seul `HistGradientBoostingRegressor` par quantile, avec
`stock_sku` en feature catégorielle native.

**Pourquoi** :
- Mutualise l'apprentissage : les SKU à faible volume profitent du
  pattern saisonnier appris sur les SKU populaires.
- Un seul modèle à entraîner, à archiver, à tuner.
- HGB gère les catégories sans one-hot.

**Inconvénient** : un SKU très atypique peut tirer le modèle dans le
mauvais sens. Mitigation : `--diagnose` permet de l'identifier.

### Quantile regression plutôt que prédiction ponctuelle + bootstrap

**Choix** : 3 modèles HGB indépendants avec `loss="quantile"`.

**Pourquoi** :
- Natif scikit-learn, pas de dépendance lourde.
- Rapide : < 30 s pour ~400 lignes × 3 quantiles.
- Les quantiles sont triés par ligne pour garantir
  `q_low ≤ q_med ≤ q_high` (croisement de quantiles possible avec 3
  modèles indépendants).

**Inconvénient** : pas de modèle conjoint, pas de cohérence par
construction. Acceptable pour notre usage.

### Forecast itératif vs. multi-output

**Choix** : prédiction itérative N-step où le `q_med` prédit à t devient
le `lag_1` de t+1.

**Pourquoi** :
- Permet de prévoir 26 semaines même si on n'a entraîné que sur des
  cibles à 1 semaine.
- Cohérent avec la simulation Monte-Carlo de la rupture.

**Inconvénient** : l'incertitude propagée est sous-estimée (on
réinjecte le médian, pas un échantillon). Mitigation : la simulation
Monte-Carlo aval échantillonne la distribution complète.

### Best-effort partout

À chaque point d'intégration, un échec ML laisse passer la baseline
sans bruit visible :
- `_persist_weekly_history()` : `try/except`, log warning.
- `_attach_ml_if_enabled()` : `try/except`, log warning.
- `train_global_model()` : retourne `None` si historique insuffisant.
- `attach_ml_projections()` : skip silencieusement les SKU sans assez
  d'historique.

L'utilisateur n'est jamais bloqué.

---

## Tests

Le sous-système ML a sa propre suite (~110 tests) :

| Fichier | Couverture |
|---|---|
| `test_ml_dataset.py` | Persistance idempotente, schéma, merge, filtrage |
| `test_ml_features.py` | Calendrier (jours fériés FR), anti-leakage lags + rolling |
| `test_ml_model.py` | Ridge baseline : fit/predict, save/load, SKU inconnu |
| `test_ml_projection.py` | Quantile HGB, sampling Monte-Carlo, dates de rupture |
| `test_ml_evaluation.py` | MAPE/pinball/coverage, walk-forward, règle de promotion |
| `test_ml_registry.py` | Archive, current symlink, journal CSV, drift |
| `test_ml_inference.py` | Orchestrateur attach_ml_projections, edge cases |
| `test_ml_pdf_smoke.py` | Génération PDF avec et sans `ml_projection` |
| `test_ml_persistence_integration.py` | Branchement dans `run_stock_report` |
| `test_ml_config.py` | Load/save/roundtrip, défauts, validation |
| `test_ml_diagnose.py` | Métriques par SKU, formatage table |
| `test_ml_tuning.py` | Halving (échantillon/exhaustif), garde-fou de persistance |

Lancement :

```bash
# Tous les tests ML
python -m pytest tests/test_ml_*.py -v

# Suite complète du projet
python -m pytest tests/
```

Couverture attendue :

```bash
python -m pytest --cov=stocks.ml tests/
```

Tous les tests utilisent `tmp_path` pour isoler les artefacts ; rien
n'est jamais écrit dans `stocks/data/` ou `stocks/models/` réels.
