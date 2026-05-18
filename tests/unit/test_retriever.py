# test_retriever.py

from rag.vectorstore import create_vectorstore
from rag.embeddings import embedding_model
from rag.retriever import get_retriever, search_similar_chunks

import pytest


@pytest.fixture
def sample_vectorstore():

    chunks = [
        "machine learning basics",
        "deep learning",
        "transformers",
        "cooking recipe"
    ]

    return create_vectorstore(
        chunks,
        video_id="test",
        embedding_model=embedding_model
    )


def test_retriever_top_k(sample_vectorstore):

    retriever = get_retriever(
        sample_vectorstore,
        k=2
    )

    results = search_similar_chunks(
        retriever,
        "artificial intelligence"
    )

    assert len(results) == 2


def test_retriever_returns_metadata(
    sample_vectorstore
):

    retriever = get_retriever(
        sample_vectorstore,
        k=1
    )

    results = search_similar_chunks(
        retriever,
        "machine learning"
    )

    result = results[0]

    assert "video_id" in result
    assert "chunk_id" in result

    
def test_retriever_exception_handling():

    class FakeRetriever:

        def invoke(self, query):
            raise Exception("Retriever crashed")

    results = search_similar_chunks(
        FakeRetriever(),
        "machine learning"
    )

    assert results == []