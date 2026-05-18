# run_rag.py
import time
from run_pipeline import run_pipeline


def run_rag(test_case, config, cache, llm):

    start = time.time()

    try:
        result = run_pipeline(
            youtube_url=test_case["youtube_url"],
            user_query=test_case["task"],
            cache=cache,
            llm=llm,
            config=config
        )

        if not result:
            return {
                "output": "",
                "chunks": [],
                "processed_chunks": [],
                "latency": 0.0
            }

        output = result.get("output", "")

        chunks = result.get("chunks", []) or []

        processed_chunks = (
            result.get("processed_chunks", []) or []
        )

    except Exception:
        return {
            "output": "",
            "chunks": [],
            "processed_chunks": [],
            "latency": 0.0
        }

    latency = time.time() - start

    return {
        "output": output,
        "chunks": chunks,
        "processed_chunks": processed_chunks,
        "latency": latency
    }