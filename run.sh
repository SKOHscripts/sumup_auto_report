#!/bin/bash
# Point d'entrée unique pour crontab et exécution manuelle.
#
# Usage : ./run.sh <module> [options python]
#   modules : stocks | adhesions | paheko
#
# Exemples crontab :
#   0 9 * * 1  /home/skoh/.../sumup/run.sh stocks
#   0 8 * * *  /home/skoh/.../sumup/run.sh adhesions
#   0 7 1 * *  /home/skoh/.../sumup/run.sh paheko

set -euo pipefail

# Toujours s'exécuter depuis la racine du projet (nécessaire pour python -m)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STATE_DIR="$SCRIPT_DIR/.state"
mkdir -p "$STATE_DIR"

PYTHON="${PYTHON:-/usr/bin/python3}"

usage() {
    cat <<EOF
Usage: $0 <module> [options]

Modules :
  stocks     Rapport hebdomadaire des stocks (git pull/push automatique)
  adhesions  Rapport des adhésions (garde-fou 7 jours entre les runs)
  paheko     Tableau de bord Paheko
  purchases  Intègre les achats depuis Google Drive dans stock_items.json

Les options supplémentaires sont transmises au script Python.
  $0 stocks --no-mail
  $0 adhesions --start 2026-01-01 --end 2026-03-31
  $0 paheko
  $0 purchases
  $0 purchases --dry-run
  $0 purchases --local /chemin/vers/ACHATS_suivi_stock.xlsx
EOF
    exit 1
}

[ $# -lt 1 ] && usage

MODULE="$1"
shift

case "$MODULE" in

    stocks)
        # Commit les éventuels changements de stock_items.json du run précédent
        if [ -n "$(git status --porcelain stocks/stock_items.json)" ]; then
            git add stocks/stock_items.json
            git commit -m "auto: maj des stocks suite à l'exécution précédente"
        fi

        # Pull avec priorité au remote en cas de conflit
        git pull --rebase -X theirs origin master

        # Mise à jour des stocks depuis le fichier Excel (Google Drive)
        "$PYTHON" -m stocks.update_stock_from_purchases

        # Rapport hebdomadaire + recalage local (décompte des ventes)
        "$PYTHON" -m stocks.sumup_stocks "$@"

        # Commit + push si stock_items.json a été modifié (achats + décompte ventes)
        if [ -n "$(git status --porcelain stocks/stock_items.json)" ]; then
            git add stocks/stock_items.json
            git commit -m "auto: stocks au $(date +'%Y-%m-%d %H:%M') (achats + décompte ventes)"
            git push origin master
        fi
        ;;

    adhesions)
        STATE_FILE="$STATE_DIR/.last_adhesions_run"
        TODAY=$(date +%F)
        IS_LAST_DAY=$([ "$(date -d tomorrow +%d)" = "01" ] && echo "yes" || echo "no")

        LAST_RUN=""
        [ -f "$STATE_FILE" ] && LAST_RUN=$(cat "$STATE_FILE")

        DAYS_SINCE=999
        if [ -n "$LAST_RUN" ]; then
            LAST_EPOCH=$(date -d "$LAST_RUN" +%s)
            TODAY_EPOCH=$(date -d "$TODAY" +%s)
            DAYS_SINCE=$(( (TODAY_EPOCH - LAST_EPOCH) / 86400 ))
        fi

        if [ "$DAYS_SINCE" -ge 7 ] || [ "$IS_LAST_DAY" = "yes" ]; then
            START="${LAST_RUN:-$(date -d '7 days ago' +%F)}"
            END="$TODAY"
            echo "[$(date)] Lancement adhésions $START → $END"
            "$PYTHON" -m adhesions.sumup_adhesions --start "$START" --end "$END" "$@"
            echo "$TODAY" > "$STATE_FILE"
        else
            echo "[$(date)] Skip — dernier run il y a $DAYS_SINCE j ($LAST_RUN)"
        fi
        ;;

    paheko)
        "$PYTHON" -m paheko_stats.paheko "$@"
        ;;

    purchases)
        # Pull avec priorité au remote en cas de conflit
        git pull --rebase -X theirs origin master

        # Intégration des achats
        "$PYTHON" -m stocks.update_stock_from_purchases "$@"

        # Commit + push si stock_items.json a été modifié
        if [ -n "$(git status --porcelain stocks/stock_items.json)" ]; then
            git add stocks/stock_items.json
            git commit -m "auto: integration achats au $(date +'%Y-%m-%d %H:%M')"
            git push origin master
        fi
        ;;

    *)
        echo "Erreur : module inconnu '$MODULE'" >&2
        usage
        ;;
esac
