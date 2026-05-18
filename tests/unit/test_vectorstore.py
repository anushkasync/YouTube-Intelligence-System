# test_vectorstore.py

import pytest

from rag.vectorstore import create_vectorstore
from rag.embeddings import embedding_model


@pytest.fixture
def sample_vectorstore():

    chunks = [
        "machine learning basics",
        "deep learning models",
        "transformers in ai"
    ]

    return create_vectorstore(
        chunks,
        video_id="test_video",
        embedding_model=embedding_model
    )


def test_vectorstore_creation(sample_vectorstore):

    assert sample_vectorstore is not None


def test_vectorstore_similarity_search(sample_vectorstore):

    results = sample_vectorstore.similarity_search(
        "artificial intelligence",
        k=2
    )

    assert len(results) == 2


def test_vectorstore_metadata(sample_vectorstore):

    results = sample_vectorstore.similarity_search(
        "machine learning",
        k=1
    )

    doc = results[0]

    assert doc.metadata["video_id"] == "test_video"

def test_vectorstore_empty_chunks():

    vectorstore = create_vectorstore(
        chunks=[],
        video_id="test_video",
        embedding_model=embedding_model
    )

    assert vectorstore is None

def test_vectorstore_exception_handling():

    class FakeEmbeddingModel:

        def embed_documents(self, texts):
            raise Exception("Embedding failed")

    vectorstore = create_vectorstore(
        chunks=["machine learning"],
        video_id="test_video",
        embedding_model=FakeEmbeddingModel()
    )

    assert vectorstore is None