
from rag.embeddings import embedding_model

def test_embeddings_created_for_all_chunks():

    chunks = [
        "This is chunk one",
        "This is chunk two",
        "This is chunk three"
    ]

    embeddings = embedding_model.embed_documents(chunks)

    assert len(embeddings) == len(chunks)


def test_embedding_dimensions_consistent():

    chunks = [
        "hello world",
        "machine learning",
        "rag systems"
    ]

    embeddings = embedding_model.embed_documents(chunks)

    dims = [len(vec) for vec in embeddings]

    assert len(set(dims)) == 1

def test_empty_embedding_input():

    embeddings = embedding_model.embed_documents([])

    assert embeddings == []
    