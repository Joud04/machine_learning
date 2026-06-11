"""Accès à l'index vectoriel Qdrant.

Contrat d'interface (pour R3) :
    store = QdrantStore(QDRANT_URL, COLLECTION_NAME)
    store.ensure_collection()
    store.upsert([{"id": ..., "vector": [...], "payload": {"text": ..., "metadata": {...}}}])
    store.search(vector, top_k)  -> [{"score": float, "text": str, "metadata": dict}]
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import COLLECTION_NAME, EMBED_DIM, QDRANT_URL


class QdrantStore:
    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION_NAME):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self) -> None:
        """Crée la collection (cosinus, dim 384) si elle n'existe pas déjà."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def upsert(self, points: list[dict]) -> None:
        """Insère/met à jour des points {id, vector, payload}.

        L'id étant déterministe (uuid5 du chunk), ré-indexer ne crée pas de
        doublon : le point existant est écrasé (idempotence).
        """
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        """Recherche les top_k plus proches voisins du vecteur requête."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
            }
            for point in result.points
        ]

    def count(self) -> int:
        """Nombre de points indexés (utile pour vérifier l'idempotence)."""
        return self.client.count(collection_name=self.collection).count
