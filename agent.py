from modules import generate_summary, generate_keypoints, generate_questions
from guardrails.input_guard import validate_query
from guardrails.context_guard import FALLBACK_RESPONSE, validate_context
from guardrails.output_guard import validate_output
from prompts.agent_intent_prompt import INTENT_PROMPT
from rag.embeddings import embedding_model
from guardrails.intent_guard import VALID_TASKS
from logger import get_logger

def classify_intent(query, llm, metadata=None):

    prompt = INTENT_PROMPT.format(query=query)
    response = llm.invoke(prompt).content.strip().lower()

    if response not in VALID_TASKS:
        if metadata:
            metadata["failure_reason"] = "INVALID_INTENT"
            metadata["fallback_triggered"] = True
        return None

    return response


# MODE SELECTOR

def decide_mode(chunks):
    if len(chunks) <= 5:
        return "small"
    elif len(chunks) < 20:
        return "medium"
    return "long"


# RAG QA (retrieval based)

def answer_with_rag(query, vectorstore, llm, metadata=None, k=3):

    results = vectorstore.similarity_search(query, k=k)

    if not results:
        if metadata:
            metadata["retrieval_used"] = True
            metadata["fallback_triggered"] = True
            metadata["failure_reason"] = "NO_RETRIEVED_CHUNKS"
            metadata["retrieval"]["score"] = 0.0
            metadata["retrieval"]["chunks"] = []
            metadata["retrieval"]["top_k"] = 0

        return FALLBACK_RESPONSE

    query_emb = embedding_model.embed_query(query)

    chunk_embs = embedding_model.embed_documents(
        [r.page_content for r in results]
    )

    valid, fallback, score = validate_context(query_emb, chunk_embs)

    if metadata:
        metadata["retrieval_used"] = True
        metadata["retrieval"]["score"] = score
        metadata["retrieval"]["chunks"] = [r.page_content for r in results]
        metadata["retrieval"]["top_k"] = len(results)

    if fallback:
        if metadata:
            metadata["fallback_triggered"] = True
            metadata["failure_reason"] = "LOW_CONTEXT_RELEVANCE"

        return FALLBACK_RESPONSE
    
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content.strip()


def run_agent(
    task,
    query,
    chunks,
    processed_chunks,
    vectorstore,
    llm,
    return_meta=False,
    trace_id=None
):

    logger = get_logger(trace_id)

    logger.info("Agent started")

    mode = decide_mode(chunks)

    metadata = {
        "task": task,
        "mode": mode,
        "retrieval_used": False,
        "fallback_triggered": False,
        "failure_reason": None,
        "retrieval": {
            "score": 0.0,
            "chunks": [],
            "top_k": 0
        },
        "latency": None
    }

    result = None

    if task == "summary":

        logger.info("Running summary module")

        result = generate_summary(
            processed_chunks,
            llm,
            mode
        )

    elif task == "keypoints":

        logger.info("Running keypoints module")

        result = generate_keypoints(
            processed_chunks,
            llm,
            mode
        )

    elif task == "qa_gen":

        logger.info("Running question generation module")

        result = generate_questions(
            processed_chunks,
            llm,
            mode
        )

    elif task == "rag":

        logger.info("Running RAG module")

        result = answer_with_rag(
            query,
            vectorstore,
            llm,
            metadata
        )

    if not validate_output(result):

        logger.warning("Invalid output")

        metadata["failure_reason"] = "OUTPUT_INVALID"

        metadata["fallback_triggered"] = True

        result = FALLBACK_RESPONSE

    if return_meta:

        return {
            "output": result,
            "metadata": metadata
        }

    return result