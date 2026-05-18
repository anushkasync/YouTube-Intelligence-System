# run_evaluations.py 
from evaluations.llm_as_judge import llm_judge_ux
from evaluations.ragas_metrics import ragas_metrics
from metrics.retrieval_eval import retrieval_metrics


def run_evaluation(query, output, processed_chunks, latency, config):
    
    retrieval = retrieval_metrics(query, processed_chunks)
    context = processed_chunks if processed_chunks else []
    ragas = ragas_metrics(query, context, output)
    ux = llm_judge_ux(output, context, query)

    ragas_score = (
        ragas.get("faithfulness", 0.5) +
        ragas.get("answer_relevancy", 0.5)
    ) / 2

    # -----------------------------
    # LLM JUDGE (UX)
    # -----------------------------

    ux_score = ux.get("overall_score", 0.5)

    generation_quality = (
        0.6 * ragas_score +
        0.4 * ux_score
    )

    return {
        "ragas_score": ragas_score,
        "ux_score": ux_score,
        "generation_quality": generation_quality,
        "latency": latency
    }