"""Module ML pour la projection de rupture de stocks.

Sous-modules :
  - dataset    : persistance de l'historique hebdomadaire en parquet
  - features   : feature engineering (calendrier, lags, saisonnalité)
  - model      : entraînement / inférence des modèles quantile multi-SKU
  - projection : simulation Monte-Carlo de la date de rupture
  - evaluation : backtest walk-forward et détection de drift
  - registry   : sauvegarde/chargement des modèles + métadonnées
"""
