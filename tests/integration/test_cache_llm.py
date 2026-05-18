import pytest

from utils.cache_manager import CacheManager
from utils.llm import OpenRouterLLM


# ---------------------------------------------------
# MOCK LLM
# ---------------------------------------------------

class MockResponse:

    def __init__(self, content):
        self.content = content


class MockLLM:

    def __init__(self):

        self.call_count = 0

    def invoke(self, prompt):

        self.call_count += 1

        return MockResponse(
            "fresh response from LLM"
        )

def fake_post(*args, **kwargs):

    class FakeResponse:

        def __init__(self):

            self.status_code = 200

        def json(self):

            return {
                "choices": [
                    {
                        "message": {
                            "content": "fresh response from LLM"
                        }
                    }
                ]
            }

    return FakeResponse()

# ---------------------------------------------------
# TEST: CACHE INTEGRATION FLOW
# ---------------------------------------------------

def test_llm_cache_integration(tmp_path, monkeypatch):

    cache = CacheManager(base_dir=tmp_path)

    def fake_post(*args, **kwargs):

        class FakeResponse:

            def __init__(self):

                self.status_code = 200

            def json(self):

                return {
                    "choices": [
                        {
                            "message": {
                                "content": "fresh response from LLM"
                            }
                        }
                    ]
                }

        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    service = OpenRouterLLM(
        api_key="fake",
        model="test-model",
        cache_manager=cache
    )

    res1 = service.invoke("What is machine learning?")

    assert res1.content == "fresh response from LLM"

    # cache should now exist
    key = cache.make_llm_key(
        "What is machine learning?",
        "test-model"
    )

    assert cache.get_llm_output(key) == "fresh response from LLM"

    # second call → cache hit (no API call)
    res2 = service.invoke("What is machine learning?")

    assert res2.content == "fresh response from LLM"


# ---------------------------------------------------
# CACHE PERSISTENCE TEST
# ---------------------------------------------------

def test_cache_persistence(tmp_path):

    cache1 = CacheManager(base_dir=tmp_path)

    key = cache1.make_llm_key("hello", "model")

    cache1.save_llm_output(key, "cached value")

    # simulate restart (new instance)
    cache2 = CacheManager(base_dir=tmp_path)

    assert cache2.get_llm_output(key) == "cached value"