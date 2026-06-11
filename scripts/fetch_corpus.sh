#!/usr/bin/env bash
# Recuperation du corpus public (projet A : profil "open" = data.gouv.fr)
# Usage : PROFILE=open DATA_QUERY="intelligence artificielle" N_DOCS=10 bash scripts/fetch_corpus.sh
set -euo pipefail
PROFILE="${PROFILE:-open}"
DATA_QUERY="${DATA_QUERY:-intelligence artificielle}"
N_DOCS="${N_DOCS:-10}"

if [ "$PROFILE" = "open" ]; then
  python scripts/fetch_datagouv.py "$DATA_QUERY" "$N_DOCS" "corpus/raw/datagouv"
else
  echo "PROFILE inconnu pour le projet A : $PROFILE (utiliser 'open')" >&2
  exit 1
fi
