from evaluations.test_loader import load_test_cases
from experiments.run_rag import run_rag
from evaluations.run_evaluations import run_evaluation
from benchmarks.run_benchmark import run_benchmark


def run_full_suite(cache, llm, config, mode="full"):
    """
    mode:
        - benchmark → NO LLM (only retrieval + system scoring)
        - eval → LLM + RAGAS evaluation only
        - full → both benchmark + evaluation
    """

    test_cases = load_test_cases()
    pipeline_results = []

    eval_results = []

    for test in test_cases:

        run_result = run_rag(test, config, cache, llm)

        output = run_result.get("output", "")
        processed_chunks = run_result.get("processed_chunks", []) or []
        latency = run_result.get("latency", 0.0)

        base = {
            "test_id": test.get("id"),
            "task": test.get("task"),
            "output": output,
            "processed_chunks": processed_chunks,
            "latency": latency
        }

        pipeline_results.append(base)

    if mode in ["eval", "full"]:
        for item in pipeline_results:

            eval_result = run_evaluation(
                query=item["task"],
                output=item["output"],
                processed_chunks=item["processed_chunks"],
                latency=item["latency"],
                config=config
            )

            eval_results.append({
                "test_id": item["test_id"],
                "task": item["task"],
                **eval_result
            })

    benchmark_results = None

    if mode in ["benchmark", "full"]:
        benchmark_results = run_benchmark(pipeline_results)

    return {
        "mode": mode,
        "benchmark": benchmark_results,
        "evaluations": eval_results
    }