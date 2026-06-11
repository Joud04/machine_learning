from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieve import RetrievalService


def test_retrieval_threshold_filtering():
    """Test that RetrievalService filters out documents below the SIM_THRESHOLD."""
    # Mock VectorStore
    mock_store = MagicMock()
    # Setup similarity_search_with_score to return one high and one low score
    mock_store.similarity_search_with_score.return_value = [
        (Document(page_content="High similarity doc", metadata={"source": "s1"}), 0.9),
        (Document(page_content="Low similarity doc", metadata={"source": "s2"}), 0.1),
    ]

    service = RetrievalService(vector_store=mock_store, embeddings=MagicMock())

    results = service.retrieve("test query")

    assert len(results) == 1
    assert results[0].page_content == "High similarity doc"


def test_retrieve_with_scores_returns_pairs():
    """Test that retrieve_with_scores exposes raw (doc, score) pairs for the API."""
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [
        (Document(page_content="doc", metadata={"source": "s1"}), 0.42),
    ]

    service = RetrievalService(vector_store=mock_store, embeddings=MagicMock())

    results = service.retrieve_with_scores("test query", k=3)

    assert results[0][1] == 0.42
    mock_store.similarity_search_with_score.assert_called_once_with("test query", k=3)
