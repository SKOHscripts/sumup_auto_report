#!/bin/bash
set -euo pipefail

STATE_FILE="/home/skoh/SynologyDrive/Documents/Scripts/sumup/adhesions/.last_sumup_adhesions"
TODAY=$(date +%F)                                  # ex: 2026-04-17
TOMORROW=$(date -d tomorrow +%F)
IS_LAST_DAY=$([ "$(date -d tomorrow +%d)" = "01" ] && echo "yes" || echo "no")

# Lire la date du dernier run
LAST_RUN=""
[ -f "$STATE_FILE" ] && LAST_RUN=$(cat "$STATE_FILE")

# Calculer le nombre de jours depuis le dernier run
DAYS_SINCE=999
if [ -n "$LAST_RUN" ]; then
    LAST_EPOCH=$(date -d "$LAST_RUN" +%s)
    TODAY_EPOCH=$(date -d "$TODAY" +%s)
    DAYS_SINCE=$(( (TODAY_EPOCH - LAST_EPOCH) / 86400 ))
fi

# Décision de lancement
if [ "$DAYS_SINCE" -ge 7 ] || [ "$IS_LAST_DAY" = "yes" ]; then
    # Calculer la fenêtre de 7 jours
    START=$(date -d "$TODAY - 6 days" +%F)
    END="$TODAY"

    echo "[$(date)] Lancement stats $START → $END"
    /usr/bin/python3 -m adhesions.sumup_adhesions --start "$START" --end "$END"

    # Sauvegarder la date du run
    echo "$TODAY" > "$STATE_FILE"
else
    echo "[$(date)] Skip — dernier run il y a $DAYS_SINCE j ($LAST_RUN)"
fi
