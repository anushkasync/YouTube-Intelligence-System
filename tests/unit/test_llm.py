import pytest
import requests

from utils.llm import OpenRouterLLM

# =====================================================
# MOCK RESPONSE
# =====================================================

class MockHTTPResponse:

    def __init__(self, status_code, data):

        self.status_code = status_code
        self._data = data

    def json(self):

        return self._data


# =====================================================
# MOCK CACHE MANAGER
# =====================================================

class MockCacheManager:

    def __init__(self):

        self.saved = {}

    def make_llm_key(self, prompt, model):

        return f"{prompt}-{model}"

    def get_llm_output(self, key):

        return self.saved.get(key)

    def save_llm_output(self, key, output):

        self.saved[key] = output


# =====================================================
# TEST: SUCCESSFUL API CALL
# =====================================================

def test_llm_success(monkeypatch):

    def fake_post(*args, **kwargs):

        return MockHTTPResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hello from LLM"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        requests,
        "post",
        fake_post
    )

    llm = OpenRouterLLM(
        api_key="fake_key",
        model="test-model"
    )

    response = llm.invoke("hello")

    assert response.content == "Hello from LLM"


# =====================================================
# TEST: API FAILURE
# =====================================================

def test_llm_api_failure(monkeypatch):

    def fake_post(*args, **kwargs):

        return MockHTTPResponse(
            500,
            {
                "error": "Server failed"
            }
        )

    monkeypatch.setattr(
        requests,
        "post",
        fake_post
    )

    llm = OpenRouterLLM(
        api_key="fake_key",
        model="test-model"
    )

    with pytest.raises(Exception):

        llm.invoke("hello")

# =====================================================
# TEST: CACHE HIT
# =====================================================

def test_llm_cache_hit(monkeypatch):

    cache = MockCacheManager()

    key = cache.make_llm_key(
        "hello",
        "test-model"
    )

    cache.save_llm_output(
        key,
        "cached response"
    )

    def fake_post(*args, **kwargs):

        pytest.fail("API should not be called during cache hit")

    monkeypatch.setattr(
        requests,
        "post",
        fake_post
    )

    llm = OpenRouterLLM(
        api_key="fake_key",
        model="test-model",
        cache_manager=cache
    )

    response = llm.invoke("hello")

    assert response.content == "cached response"

# =====================================================
# TEST: CACHE MISS + SAVE
# =====================================================

def test_llm_cache_miss_and_save(monkeypatch):

    cache = MockCacheManager()

    def fake_post(*args, **kwargs):

        return MockHTTPResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "fresh response"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        requests,
        "post",
        fake_post
    )

    llm = OpenRouterLLM(
        api_key="fake_key",
        model="test-model",
        cache_manager=cache
    )

    response = llm.invoke("hello")

    key = cache.make_llm_key(
        "hello",
        "test-model"
    )

    assert response.content == "fresh response"

    assert cache.saved[key] == "fresh response"
    
# =====================================================
# TEST: THROTTLE EXECUTION
# =====================================================

def test_llm_throttle_called(monkeypatch):

    called = {
        "throttle": False
    }

    def fake_throttle():

        called["throttle"] = True

    def fake_post(*args, **kwargs):

        return MockHTTPResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        requests,
        "post",
        fake_post
    )

    llm = OpenRouterLLM(
        api_key="fake_key",
        model="test-model"
    )

    monkeypatch.setattr(
        llm,
        "_throttle",
        fake_throttle
    )

    llm.invoke("hello")

    assert called["throttle"] is True