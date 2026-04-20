#!/bin/bash
# run_sumup_stocks.sh

# Arrêter l'exécution en cas d'erreur
set -e

DIR="/home/skoh/SynologyDrive/Documents/Scripts/sumup"
cd "$DIR"

# 1. Optionnel mais recommandé : On commit les changements du précédent passage
# S'il y a eu des modifications sur le json par le script lors du précédent run
if [ -n "$(git status --porcelain stock_items.json)" ]; then
    git add stock_items.json
    git commit -m "auto: maj des stocks suite à l'exécution précédente"
fi

# 2. Récupération des nouveautés distantes (avec fusion si conflit)
# On privilégie la version distante en cas de conflit bloquant (theirs)
# car de toute façon le script Python va recalculer les déductions sur la base du remote
git pull --rebase -X theirs origin master

# 3. Exécution du script Python
/usr/bin/python3 -m stocks.sumup_stocks

# 4. On commite et on pousse la nouvelle version fraîchement calculée pour ne rien perdre
if git diff --name-only HEAD | grep -qE '/stock_items\.json$|^stock_items\.json$'; then
    git add -A
    git commit -m "auto: recalage du stock au $(date +'%Y-%m-%d %H:%M')"
    git push origin master
fi
