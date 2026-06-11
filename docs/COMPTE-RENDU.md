# Compte rendu — AssistKB Search (Projet A)

> Chaque membre rédige sa propre section sous son rôle.

## R2 — Embeddings + index Qdrant

### Objectif
Vectoriser les chunks produits par R1 et les indexer dans Qdrant, en exposant
une interface de recherche stable pour R3.

### Réalisation
- **`app/embed.py`** — vectorisation avec `all-MiniLM-L6-v2`
  (`sentence-transformers`), par batch de 64, vecteurs **normalisés**
  (`normalize_embeddings=True`, cohérent avec la distance cosinus).
  - `embed_texts(texts)` : vectorisation par lot.
  - `embed_query(text)` : vectorisation d'une requête unique (utilisé par R3).
  - `python -m app.embed` : lit `data/chunks.jsonl` et indexe l'ensemble.
- **`app/store.py`** — classe `QdrantStore` :
  - `ensure_collection()` : crée la collection en distance **cosinus**,
    dimension **384**, uniquement si elle n'existe pas (ré-exécution sûre).
  - `upsert(points)` : insertion/mise à jour. L'`id` de chaque point est
    l'`uuid5` déterministe du chunk → ré-indexer **écrase** au lieu de
    dupliquer (idempotence).
  - `search(vector, top_k)` : renvoie `[{score, text, metadata}]`.
  - `count()` : nombre de points indexés.

### Contrat d'interface (pour R3)
```python
store.search(vector, top_k)  # -> [{"score": float, "text": str, "metadata": dict}]
```

### Vérifications
| Vérification | Attendu | Obtenu |
|---|---|---|
| `data/chunks.jsonl` (après `app.ingest`) | 687 chunks | ✅ 687 |
| `points_count` (`/collections/rag_chunks`) | 687 | ✅ 687 |
| Dimension du vecteur | 384 | ✅ 384 |
| Distance | Cosine | ✅ Cosine |
| Idempotence (relancer `app.embed`) | inchangé | ✅ 687 → 687 |
| `search()` (requête de test) | format `{score, text, metadata}` | ✅ conforme |

### Commandes de reproduction
```bash
docker run -d -p 6333:6333 qdrant/qdrant
DATA_QUERY="intelligence artificielle" N_DOCS=5 bash scripts/fetch_corpus.sh
python -m app.ingest        # -> data/chunks.jsonl (687 chunks)
python -m app.embed         # vectorisation + indexation
curl http://localhost:6333/collections/rag_chunks   # points_count = 687
```
