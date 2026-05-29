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

def decide_mode(chunks):
    if len(chunks) <= 5:
        return "small"
    elif len(chunks) < 20:
        return "medium"
    return "long"

def answer_with_rag(query, vectorstore, llm, metadata=None, k=4):

    results = vectorstore.similarity_search(query, k=k)

    if not results:
        if metadata:
            metadata.update({
                "retrieval_used": True,
                "fallback_triggered": True,
                "failure_reason": "NO_RETRIEVED_CHUNKS"
            })

        return FALLBACK_RESPONSE

    chunks = [doc.page_content for doc in results]

    if metadata:
        metadata.update({
            "retrieval_used": True,
            "retrieval": {
                "chunks": chunks,
                "top_k": len(chunks)
            }
        })

    context = "\n\n".join(chunks)

    prompt = f"""
You are a precise retrieval QA assistant.

Answer ONLY using the context below.

If the context partially answers the question,
provide the closest relevant answer.

Only say:
"Not found in the video."
if there is truly no relevant information.

Keep answers concise.

Context:
{context}

Question:
{query}

Answer:
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

    elif task == "rag_qa":

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