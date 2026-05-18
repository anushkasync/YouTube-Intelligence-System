from datetime import datetime
from metrics.retrieval_eval import retrieval_metrics
from metrics.system_metrics import system_metrics
from config.config import CONFIG


def run_benchmark(evaluated_results):

    print("BENCHMARK STARTED")

    if not isinstance(evaluated_results, list):
        raise ValueError("Expected list of evaluated results")

    results = {
        "timestamp": None,
        "model": CONFIG["LLM_MODEL"],
        "cases": [],
        "avg_score": 0.0
    }

    total_score = 0.0

    for item in evaluated_results:

        output = item.get("output", "")
        processed_chunks = item.get("processed_chunks", [])
        latency = item.get("latency", 0.0)

        retrieval = retrieval_metrics(
            item.get("task", ""),
            processed_chunks
        )

        system = system_metrics(latency, output)

        score = (
            0.6 * retrieval.get("average_top_k_cosine_similarity", 0.0)
            + 0.4 * (1 / (1 + latency))
        )

        total_score += score

        results["cases"].append({
    "test_id": item.get("test_id"),
    "retrieval_score": retrieval.get("average_top_k_cosine_similarity", 0.0),
    "latency": round(latency, 3),
    "system_score": round(1 / (1 + latency), 4),
    "final_score": round(score, 4)
})

    results["avg_score"] = round(
    sum(c["final_score"] for c in results["cases"]) / len(results["cases"]),
    4
)
    results["timestamp"] = datetime.utcnow().isoformat() + "Z"

    print("BENCHMARK COMPLETE")

    return results