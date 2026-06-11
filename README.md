# AssistKB Search — Projet A (RAG + Qdrant)

Pipeline RAG avec seuil de refus anti-hallucination : corpus public data.gouv.fr + base de connaissances interne (seed), indexés dans Qdrant, interrogés via une API REST qui répond en citant ses sources.

## Prérequis

- Python 3.11+
- Docker Desktop
- Une clé API Groq gratuite (console.groq.com)

## Installation

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
```

## État d'avancement

| Étage | Statut | Fichiers |
|-------|--------|----------|
| R1 — Ingestion (extraction, chunking, métadonnées) | ✅ | `app/ingest.py`, `scripts/` |
| R2 — Embeddings + index Qdrant | 🔜 | `app/embed.py`, `app/store.py` |
| R3 — Retrieval, génération LLM, API | 🔜 | `app/retrieve.py`, `app/generate.py`, `app/api.py` |
| R4 — docker-compose, métriques | 🔜 | `docker-compose.yml`, `app/metrics.py` |

## Format des chunks (`data/chunks.jsonl`, un JSON par ligne)

```json
{"id": "<uuid5 déterministe>", "text": "...", "metadata": {"source": "corpus/seed/x.html", "doc_type": "html", "position": 0}}
```

## Licences des corpus

- data.gouv.fr : Licence Ouverte / Open Licence
- `corpus/seed/` : documents pédagogiques fictifs (Neosoft)
