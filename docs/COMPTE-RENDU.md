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

### Évaluation du retrieval (étapes 9 et 8.1)

**Golden dataset (`eval/golden.jsonl`)** — 10 questions de référence ancrées sur
le corpus *seed* (déterministe, donc reproductible — contrairement au corpus
data.gouv qui change à chaque téléchargement). Chaque ligne associe une question
à la source attendue : `{"question": ..., "expected_source": ...}`.

**Métrique (`eval/recall.py`)** — pour chaque question, on interroge le retrieval
(sans appel LLM) et on mesure :
- **recall@k** : la source attendue figure-t-elle dans le top-k ?
- **MRR** (Mean Reciprocal Rank) : moyenne de `1/rang` de la bonne source.

```bash
python -m eval.recall 5     # recall@5 = 0.90 (9/10), MRR = 0.900
```

**Expérimentation top-k (`scripts/experiment.py`, étape 8.1)** — on fait varier k
pour justifier le choix de `TOP_K` :

| k | recall@k | MRR | latence moy. (ms) |
|---|----------|-----|-------------------|
| 1 | 0.90 | 0.900 | 44.0 |
| 3 | 0.90 | 0.900 | 43.2 |
| 5 | 0.90 | 0.900 | 41.7 |
| 10 | 1.00 | 0.914 | 34.5 |

**Lecture des résultats.** La bonne source est presque toujours au **rang 1**
(recall@1 déjà à 0.90, MRR ≈ 0.90) : le retrieval est précis. Une seule question
(« latence cible ») n'est rattrapée qu'à un rang profond, d'où le passage de
recall@5 = 0.90 à recall@10 = 1.00 — mais le MRR ne progresse quasiment pas
(0.900 → 0.914), signe que ce gain est marginal. **Conclusion : `TOP_K=5` est le
bon compromis** — au-delà, le recall plafonne tandis que le contexte envoyé au LLM
grossit (coût et latence de génération en hausse, sans bénéfice de pertinence).

```bash
python -m scripts.experiment   # tableau comparatif top-k
```
