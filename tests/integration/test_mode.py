import pytest

from run_pipeline import run_pipeline
from config.config import CONFIG
from utils.cache_manager import CacheManager
# -----------------------------
# Shared mocks
# -----------------------------

class MockLLM:
    def invoke(self, prompt):
        return type("R", (), {"content": "mock response"})


def fake_transcript(text_size):
    return {
        "video_id": "test123",
        "text": "word " * text_size
    }

def cache(tmp_path):
    return CacheManager(base_dir=tmp_path)


@pytest.mark.parametrize(
    "text_size, expected_mode",
    [
        (10, "small"),
        (200, "medium"),
        (2000, "long"),
    ]
)
def test_pipeline_modes(monkeypatch, cache, text_size, expected_mode):

    # Mock transcript
    monkeypatch.setattr(
        "run_pipeline.get_transcript",
        lambda url: fake_transcript(text_size)
    )

    # Mock intent classifier
    monkeypatch.setattr(
        "run_pipeline.classify_intent",
        lambda query, llm, metadata: "summary"
    )

    # Force deterministic mode decision
    monkeypatch.setattr(
        "run_pipeline.decide_mode",
        lambda chunks: expected_mode
    )

    llm = MockLLM()

    result = run_pipeline(
        youtube_url="https://youtu.be/fake",
        user_query="summarize this video",
        cache=cache,
        llm=llm,
        config=CONFIG
    )

    metadata = result["metadata"]

    # -----------------------------
    # Core assertions
    # -----------------------------

    assert metadata["mode"] == expected_mode
    assert isinstance(result["output"], str)
    assert result["output"] != ""
    assert metadata["fallback_triggered"] is False