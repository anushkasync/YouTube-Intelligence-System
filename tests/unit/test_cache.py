import pytest
import json
from utils.cache_manager import CacheManager, stable_hash

@pytest.fixture
def cache(tmp_path):

    return CacheManager(base_dir=tmp_path)

def test_stable_hash_consistency():

    h1 = stable_hash({"a": 1, "b": 2, "c": 3})
    h2 = stable_hash({"b": 2, "a": 1, "c": 3})

    assert h1 == h2

def test_transcript_save_get(cache):

    cache.save_transcript("vid1", "hello transcript")

    assert cache.get_transcript("vid1") == "hello transcript"


def test_chunks_save_get(cache):

    key = cache.make_chunk_key("vid1", 10, 2)

    cache.save_chunks(key, ["a", "b"])

    assert cache.get_chunks(key) == ["a", "b"]


def test_processed_chunks(cache):

    key = cache.make_processed_key("vid1", "small")

    cache.save_processed_chunks(key, ["x", "y"])

    assert cache.get_processed_chunks(key) == ["x", "y"]


def test_llm_cache(cache):

    key = cache.make_llm_key("hello", "gpt")

    cache.save_llm_output(key, "response")

    assert cache.get_llm_output(key) == "response"


def test_vectorstore_key_generation(cache):

    key = cache.make_vectorstore_key(
        "vid1",
        "emb-model",
        10,
        2
    )

    expected_hash = stable_hash({
        "video_id": "vid1",
        "embedding_model": "emb-model",
        "chunk_size": 10,
        "overlap": 2
    })

    assert key == expected_hash

class FakeVectorstore:

    def __init__(self):
        self.saved_path = None

    def save_local(self, path):
        self.saved_path = path


class FakeEmbeddingModel:
    pass


def test_vectorstore_save(cache):

    vs = FakeVectorstore()

    key = cache.make_vectorstore_key(
        "vid1",
        "emb",
        10,
        2
    )

    cache.save_vectorstore(key, vs)

    expected_path = cache.vectorstore_path(key)

    assert vs.saved_path == expected_path