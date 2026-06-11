from typing import List, Tuple

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.config import QDRANT_URL, COLLECTION_NAME, EMBED_MODEL, SIM_THRESHOLD


class RetrievalService:
    """
    Service responsible for retrieving the most relevant documents from the vector store.
    """
    def __init__(self, vector_store=None, embeddings=None):
        # Dependency Injection for testability
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        if vector_store:
            self.vector_store = vector_store
        else:
            # Initialize Qdrant client and LangChain wrapper
            client = QdrantClient(url=QDRANT_URL)
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings,
                content_payload_key="text",
            )

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Returns (document, cosine similarity) pairs, best score first.
        Exposed separately so the API can check the best score against the
        refusal threshold without running a second search.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieves documents based on query similarity, filtering by a minimum threshold.
        """
        results = self.retrieve_with_scores(query, k=k)

        # Filter documents based on SIM_THRESHOLD
        # Note: Qdrant scores are cosine similarity (0 to 1, higher is better)
        return [doc for doc, score in results if score >= SIM_THRESHOLD]
