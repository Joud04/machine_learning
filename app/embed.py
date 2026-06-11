"""Vectorisation des chunks (all-MiniLM-L6-v2) et indexation dans Qdrant.

Usage :
    python -m app.embed        # lit data/chunks.jsonl -> index Qdrant

Idempotent : relancer ne change pas le nombre de points (ids déterministes).
"""
import json
import pathlib

from sentence_transformers import SentenceTransformer

from app.config import COLLECTION_NAME, EMBED_MODEL, QDRANT_URL
from app.store import QdrantStore

CHUNKS_PATH = pathlib.Path("data/chunks.jsonl")
BATCH_SIZE = 64

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Charge le modèle une seule fois (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Vectorise une liste de textes par batch de 64, vecteurs normalisés."""
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Vectorise une requête unique (utilisé par R3 avant store.search)."""
    return embed_texts([text])[0]


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    if not CHUNKS_PATH.exists():
        print(f"{CHUNKS_PATH} introuvable. Lancez d'abord : python -m app.ingest")
        return

    chunks = load_chunks()
    if not chunks:
        print("Aucun chunk à indexer.")
        return

    store = QdrantStore(QDRANT_URL, COLLECTION_NAME)
    store.ensure_collection()

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, show_progress=True)

    points = [
        {
            "id": chunk["id"],
            "vector": vector,
            "payload": {"text": chunk["text"], "metadata": chunk["metadata"]},
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    for start in range(0, len(points), BATCH_SIZE):
        store.upsert(points[start : start + BATCH_SIZE])

    print(f"{len(points)} vecteurs indexés dans '{COLLECTION_NAME}' "
          f"(total en base : {store.count()})")


if __name__ == "__main__":
    main()
