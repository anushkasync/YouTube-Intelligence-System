# tests/integration/test_rag_flow.py

import pytest

from agent import answer_with_rag
from guardrails.context_guard import FALLBACK_RESPONSE
from rag.vectorstore import create_vectorstore

from rag.embeddings import embedding_model


# ---------------------------------------------------
# MOCK RESPONSE
# ---------------------------------------------------

class MockResponse:

    def __init__(self, content):
        self.content = content


# ---------------------------------------------------
# SUCCESS LLM
# ---------------------------------------------------

class SuccessMockLLM:

    def invoke(self, prompt):

        # verifies retrieved context reached LLM
        assert "Machine learning uses neural networks." in prompt

        return MockResponse(
            "This is the generated RAG answer."
        )


# ---------------------------------------------------
# FALLBACK LLM
# ---------------------------------------------------

class FallbackMockLLM:

    def invoke(self, prompt):

        pytest.fail(
            "LLM should not be called during fallback"
        )


# ---------------------------------------------------
# FIXTURES
# ---------------------------------------------------

@pytest.fixture
def chunks():

    return [
        "Python is a programming language.",
        "Machine learning uses neural networks.",
        "FAISS performs similarity search.",
        "Deep learning is a subset of machine learning."
    ]


@pytest.fixture
def vectorstore(chunks):

    return create_vectorstore(
        video_id="vid1",
        chunks=chunks,
        embedding_model=embedding_model
    )


@pytest.fixture
def success_llm():

    return SuccessMockLLM()


@pytest.fixture
def fallback_llm():

    return FallbackMockLLM()


@pytest.fixture
def metadata():

    return {
        "retrieval": {}
    }


# ---------------------------------------------------
# TEST: RAG INTEGRATION SUCCESS
# ---------------------------------------------------

def test_rag_integration_success(
    vectorstore,
    success_llm,
    metadata
):

    result = answer_with_rag(
        query="What is machine learning?",
        vectorstore=vectorstore,
        llm=success_llm,
        metadata=metadata,
        k=2
    )

    assert result == "This is the generated RAG answer."

    assert metadata["retrieval"]["top_k"] > 0

    assert metadata["retrieval"]["score"] > 0


# ---------------------------------------------------
# TEST: RAG INTEGRATION FALLBACK
# ---------------------------------------------------

def test_rag_integration_fallback(
    vectorstore,
    fallback_llm,
    metadata,
    monkeypatch
):

    def fake_validate_context(
        query_emb,
        chunk_embs
    ):
        return False, True, 0.1

    monkeypatch.setattr(
        "agent.validate_context",
        fake_validate_context
    )

    result = answer_with_rag(
        query="Explain black holes in space",
        vectorstore=vectorstore,
        llm=fallback_llm,
        metadata=metadata,
        k=2
    )

    assert result == FALLBACK_RESPONSE

    assert metadata["fallback_triggered"] is True