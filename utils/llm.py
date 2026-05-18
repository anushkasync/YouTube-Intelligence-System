import time
import threading
import requests

from logger import get_logger

logger = get_logger("LLM")


class OpenRouterLLM:

    def __init__(
        self,
        api_key,
        model,
        cache_manager=None
    ):

        self.api_key = api_key
        self.model = model
        self.cache_manager = cache_manager

        self.min_interval = 3.0

        self._lock = threading.Lock()
        self._last_call = 0


    def _throttle(self):

        with self._lock:

            now = time.time()

            wait = (
                self.min_interval
                -
                (now - self._last_call)
            )

            if wait > 0:
                time.sleep(wait)

            self._last_call = time.time()


    def invoke(self, prompt):

        cache_key = None

        if self.cache_manager:

            cache_key = (
                self.cache_manager.make_llm_key(
                    prompt,
                    self.model
                )
            )

            cached = (
                self.cache_manager.get_llm_output(
                    cache_key
                )
            )

            if cached is not None:

                logger.info("LLM cache hit")

                return type(
                    "LLMResponse",
                    (),
                    {"content": cached}
                )

            logger.info(
                "LLM cache miss → calling API"
            )

        self._throttle()

        max_retries = 3

        for attempt in range(max_retries):

            try:

                logger.info(
                    f"LLM request started "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                response = requests.post(
                    url=(
                        "https://openrouter.ai/"
                        "api/v1/chat/completions"
                    ),

                    headers={
                        "Authorization":
                            f"Bearer {self.api_key}",

                        "Content-Type":
                            "application/json"
                    },

                    json={
                        "model": self.model,

                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    },

                    timeout=60
                )

                try:
                    data = response.json()

                except Exception:

                    data = {
                        "error":
                            "Invalid JSON response"
                    }

                if response.status_code == 200:

                    content = (
                        data["choices"][0]
                        ["message"]["content"]
                    )

                    break

                logger.error(
                    f"LLM API error: {data}"
                )

                raise Exception(
                    f"OpenRouter Error: {data}"
                )

            except Exception as e:

                logger.warning(
                    f"LLM request failed "
                    f"(attempt {attempt + 1}): {e}"
                )

                if attempt == max_retries - 1:
                    raise

                wait_time = 2 ** attempt

                logger.info(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

        if self.cache_manager and cache_key:

            self.cache_manager.save_llm_output(
                cache_key,
                content
            )

        return type(
            "LLMResponse",
            (),
            {"content": content}
        )