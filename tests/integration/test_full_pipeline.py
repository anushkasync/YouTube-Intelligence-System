import pytest
from run_pipeline import run_pipeline
from utils.cache_manager import CacheManager
from config.config import EMBEDDING_MODEL_NAME


def fake_get_transcript_success(youtube_url):
    return {
        "video_id": "fake123",
        "source": "youtube_transcript_api",
        "text": "Machine learning uses neural networks. Python is powerful."
    }

class MockLLM:
    def invoke(self, prompt):
        return type("R", (), {"content": "system test answer"})


class FakeEmbedding:
    def embed_documents(self, texts):
        return [[1, 1, 1] for _ in texts]

    def embed_query(self, text):
        return [1, 1, 1]


@pytest.fixture
def config():
    embedding_model_name = "thenlper/gte-small"

    return {
        "CHUNK_SIZE": 10,
        "OVERLAP": 2,
        "TOP_K": 2,
        "N_CLUSTERS": 2,
        "EMBEDDING_MODEL_NAME": embedding_model_name
    }

@pytest.fixture
def cache(tmp_path):
    return CacheManager(base_dir=tmp_path)


def test_run_pipeline_system(monkeypatch, cache, config):

    monkeypatch.setattr(
        "run_pipeline.get_transcript",
        fake_get_transcript_success
    )

    monkeypatch.setattr(
        "run_pipeline.classify_intent",
        lambda query, llm, metadata: "summary"
    )

    llm = MockLLM()

    result = run_pipeline(
        youtube_url="https://youtu.be/fake123",
        user_query="What is machine learning?",
        cache=cache,
        llm=llm,
        config=config
    )

    assert isinstance(result["output"], str)
    assert result["output"] is not None

    metadata = result["metadata"]

    assert metadata["latency"] > 0

    assert "cache" in metadata

def test_pipeline_invalid_intent(monkeypatch, cache, config):

    monkeypatch.setattr(
        "run_pipeline.get_transcript",
        fake_get_transcript_success
    )

    monkeypatch.setattr(
        "run_pipeline.classify_intent",
        lambda query, llm, metadata: None
    )

    llm = MockLLM()

    result = run_pipeline(
        youtube_url="https://youtu.be/fake123",
        user_query="random query",
        cache=cache,
        llm=llm,
        config=config
    )

    assert result["metadata"]["failure_reason"] == "INVALID_INTENT"

    assert result["metadata"]["fallback_triggered"] is True

def test_pipeline_empty_transcript(monkeypatch, cache, config):

    monkeypatch.setattr(
        "run_pipeline.classify_intent",
        lambda query, llm, metadata: "summary"
    )

    monkeypatch.setattr(
        "run_pipeline.get_transcript",
        lambda url: {
            "video_id": "fake123",
            "text": ""
        }
    )

    llm = MockLLM()

    result = run_pipeline(
        youtube_url="https://youtu.be/fake123",
        user_query="summarize",
        cache=cache,
        llm=llm,
        config=config
    )

    assert result["metadata"]["failed_stage"] == "transcript"

def test_pipeline_invalid_input(cache, config):

    llm = MockLLM()

    result = run_pipeline(
        youtube_url="https://youtu.be/fake123",
        user_query="",   # invalid empty query
        cache=cache,
        llm=llm,
        config=config
    )

    assert result["metadata"]["failure_reason"] == "INVALID_INPUT"

    assert result["metadata"]["fallback_triggered"] is True

    assert result["output"] is not None