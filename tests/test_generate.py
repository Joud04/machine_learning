from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.generate import GenerationService, REFUSAL_TEXT


def test_generation_refusal_on_empty_context():
    """Test that GenerationService returns the refusal message when no documents are provided."""
    mock_llm = MagicMock()
    service = GenerationService(llm=mock_llm)

    answer, usage = service.generate("Qu'est-ce que le RAG ?", [])

    assert answer == REFUSAL_TEXT
    assert usage == {"input_tokens": 0, "output_tokens": 0}
    # LLM should NOT have been called (refusal costs 0 token)
    mock_llm.invoke.assert_not_called()


def test_generation_invocation():
    """Test that GenerationService correctly invokes the LLM chain with documents."""
    mock_llm = MagicMock()
    service = GenerationService(llm=mock_llm)

    # Mock the chain's invoke method directly (AIMessage-like object)
    service.chain = MagicMock()
    service.chain.invoke.return_value = MagicMock(
        content="Reponse simulee [source 1]",
        usage_metadata={"input_tokens": 120, "output_tokens": 30},
    )

    docs = [Document(page_content="Le RAG combine recherche et generation.", metadata={"source": "doc1"})]
    answer, usage = service.generate("Qu'est-ce que le RAG ?", docs)

    assert answer == "Reponse simulee [source 1]"
    assert usage == {"input_tokens": 120, "output_tokens": 30}
    service.chain.invoke.assert_called_once()
