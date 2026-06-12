# AssistKB Search — Projet A (RAG + Qdrant)

Pipeline RAG avec seuil de refus anti-hallucination : corpus public data.gouv.fr + base de connaissances interne (seed), indexés dans Qdrant, interrogés via une API REST et une interface web qui répondent en citant leurs sources.

## Démarrage rapide (docker compose)

```powershell
copy .env.example .env
# renseigner GROQ_API_KEY dans .env (ne jamais committer .env)
docker compose up -d --build
```

- Interface web : http://localhost:8000/
- API : `POST http://localhost:8000/ask` — `{"question": "..."}`
- Santé : `GET http://localhost:8000/health` — Qdrant : http://localhost:6333/dashboard

L'image embarque le corpus seed ; pour indexer le corpus complet, voir « Préparer le corpus ».

## Prérequis

- Python 3.11+
- Docker Desktop
- Une clé API Groq gratuite (console.groq.com)

## Installation (développement)

```powershell
git clone https://github.com/Joud04/machine_learning.git
cd machine_learning
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# puis renseigner GROQ_API_KEY dans .env (ne jamais committer .env)
```

## Préparer le corpus

```bash
# Corpus public data.gouv.fr (le seed est déjà dans corpus/seed/)
DATA_QUERY="intelligence artificielle" N_DOCS=5 bash scripts/fetch_corpus.sh

# Extraction + chunking -> data/chunks.jsonl
python -m app.ingest

# Vectorisation + indexation dans Qdrant (compose démarré, port 6333)
python -m app.embed
```

## Exemple d'appel

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Quelle est l architecture de AssistKB v0 ?"}'
```

Réponse : `answer`, `sources` (fichier, position, score), `refused`,
`best_score`, `latency_ms`, `tokens`. Si le corpus ne contient pas
l'information, l'assistant refuse au lieu d'halluciner.

## Tests et évaluation

```bash
python -m pytest               # tests unitaires (retrieval, génération)
python test_api.py             # vérification manuelle de l'API (stack démarrée)
python -m eval.recall 5        # recall@5 / MRR sur le golden dataset
python -m scripts.experiment   # comparatif top-k (justification de TOP_K=5)
python -m app.metrics          # métriques d'exploitation (latence, refus, coût)
```

## État d'avancement

| Étage | Statut | Fichiers |
|-------|--------|----------|
| R1 — Ingestion (extraction, chunking, métadonnées) | ✅ | `app/ingest.py`, `scripts/` |
| R2 — Embeddings + index Qdrant + évaluation retrieval | ✅ | `app/embed.py`, `app/store.py`, `eval/` |
| R3 — Retrieval, génération LLM, API, UI | ✅ | `app/retrieve.py`, `app/generate.py`, `app/main.py`, `static/` |
| R4 — docker-compose, métriques | ✅ | `docker-compose.yml`, `app/metrics.py` |
| Bonus | ✅ | golden dataset + recall@k (`eval/`), reranking cross-encoder (`app/retrieve.py`) |

Compte rendu complet : [docs/COMPTE-RENDU.md](docs/COMPTE-RENDU.md).

## Format des chunks (`data/chunks.jsonl`, un JSON par ligne)

```json
{"id": "<uuid5 déterministe>", "text": "...", "metadata": {"source": "corpus/seed/x.html", "doc_type": "html", "position": 0}}
```

## Licences des corpus

- data.gouv.fr : Licence Ouverte / Open Licence
- `corpus/seed/` : documents pédagogiques fictifs (Neosoft)
