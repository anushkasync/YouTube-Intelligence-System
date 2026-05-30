import time
from transcripts import extract_video_id, get_transcript
from rag.chunker import chunk_text, Chunker
from rag.vectorstore import create_vectorstore
from rag.embeddings import embedding_model
from logger import get_logger, log_event
import uuid
from agent import run_agent, decide_mode, classify_intent
from guardrails.input_guard import validate_query
from config.policy import FALLBACK_RESPONSE

from guardrails.intent_guard import VALID_TASKS

STRATEGY_BY_MODE = {
    "small": "RAW",
    "medium": "TOP_K",
    "long": "K_MEANS",
}

def run_pipeline(
    youtube_url,
    user_query,
    cache,
    llm,
    config,
    return_chunks=False,
    task=None,
):

    trace_id = uuid.uuid4().hex
    logger = get_logger(trace_id)

    logger.info("Pipeline started")

    start = time.time()

    metadata = {
        "task": None,
        "mode": None,
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

    valid, error = validate_query(user_query)

    if not valid:

        logger.warning("Input validation failed")

        metadata["failure_reason"] = "INVALID_INPUT"
        metadata["fallback_triggered"] = True

        return {
            "output": error,
            "metadata": metadata
        }

    if task is None:
        task = classify_intent(user_query, llm, metadata)
    else:
        task = task.strip().lower()

    if task not in VALID_TASKS:
        logger.warning("Invalid task detected")

        metadata["failure_reason"] = "INVALID_INTENT"
        metadata["fallback_triggered"] = True

        return {
            "output": FALLBACK_RESPONSE,
            "metadata": metadata
        }

    metadata["task"] = task


    logger.info(f"Task classified as: {task}")
    logger.info("Fetching transcript")

    transcript_cache_hit = False

    video_id = extract_video_id(youtube_url)

    transcript = cache.get_transcript(video_id)

    if (
        not transcript
        or not transcript.get("text")
        or transcript.get("text").strip() == ""
    ):

        logger.info(
            "Transcript cache miss/invalid → fetching fresh transcript"
        )

        transcript = get_transcript(youtube_url)

        if (
            not transcript
            or not transcript.get("text")
            or transcript.get("text").strip() == ""
        ):

            logger.error("Empty transcript → stopping pipeline")

            return {
                "output": "No usable transcript found for this video.",
                "metadata": {
                    "failed_stage": "transcript",
                    "reason": transcript.get(
                        "error",
                        "empty_or_missing_text"
                    )
                }
            }

        cache.save_transcript(video_id, transcript)

    else:
        logger.info("Transcript cache hit (valid)")
        transcript_cache_hit = True

    text = transcript.get("text", "")

    cached_meta = cache.lookup_video_metadata(video_id)

    if cached_meta:
        logger.info("SQLite metadata index hit — restoring cache pointers")

    chunk_key = cache.make_chunk_key(
        video_id,
        config["CHUNK_SIZE"],
        config["OVERLAP"]
    )

    chunks = None
    chunk_cache_hit = False

    if cached_meta and cached_meta["chunk_key"] == chunk_key:
        chunks = cache.get_chunks(chunk_key)
        if chunks is not None:
            chunk_cache_hit = True
            logger.info("Chunks restored via metadata index")

    if chunks is None:
        chunks = cache.get_chunks(chunk_key)
        chunk_cache_hit = chunks is not None

    if chunks is None:

        chunks = chunk_text(
            text,
            config["CHUNK_SIZE"],
            config["OVERLAP"]
        )

        cache.save_chunks(chunk_key, chunks)

    if not chunks or len(chunks) == 0:

        logger.error("Chunking failed: empty chunks")

        return {
            "output": "Unable to process video (no usable content).",
            "metadata": {
                "failed_stage": "chunking",
                "reason": "empty_chunks"
            }
        }

    logger.info(f"Generated {len(chunks)} chunks")

    logger.info("Chunking completed")

    vs_key = cache.make_vectorstore_key(
        video_id,
        embedding_model_name=config["EMBEDDING_MODEL_NAME"],
        chunk_size=config["CHUNK_SIZE"],
        overlap=config["OVERLAP"]
    )

    logger.info("Loading vectorstore")

    vectorstore = None
    vs_cache_hit = False

    if cached_meta:
        faiss_path = cached_meta["faiss_vectorstore_path"]
        vectorstore = cache.load_vectorstore_from_path(
            faiss_path,
            embedding_model,
        )
        if vectorstore is not None:
            vs_cache_hit = True
            logger.info("Vectorstore restored via metadata index")

    if vectorstore is None:
        vectorstore = cache.load_vectorstore(
            vs_key,
            embedding_model
        )
        vs_cache_hit = vectorstore is not None

    if vectorstore is None:
        logger.info("Creating vectorstore")

        logger.warning(
        "Rebuilding vectorstore embeddings"
    )

        vectorstore = create_vectorstore(
        chunks,
        video_id,
        embedding_model
    )

    if vectorstore is None:

        logger.error(
            "Vectorstore creation failed"
        )

        return {
            "output": "Unable to build retrieval system.",
            "metadata": metadata
        }

    cache.save_vectorstore(
        vs_key,
        vectorstore
    )

    logger.info(
        "Vectorstore cached successfully"
    )

    mode = decide_mode(chunks)

    metadata["mode"] = mode

    logger.info(f"Processing chunks in {mode} mode")

    processed_key = cache.make_processed_key(
        video_id,
        mode
    )

    processed_chunks = None
    processed_cache_hit = False

    if cached_meta and cached_meta["processed_key"] == processed_key:
        processed_chunks = cache.get_processed_chunks(processed_key)
        if processed_chunks is not None:
            processed_cache_hit = True
            logger.info("Processed chunks restored via metadata index")

    if processed_chunks is None:
        processed_chunks = cache.get_processed_chunks(processed_key)
        if processed_chunks is not None:
            processed_cache_hit = True

    if processed_chunks is None:

        chunker = Chunker(embedding_model)

        processed_chunks = chunker.get_processed_chunks(
            chunks,
            mode=mode,
            vectorstore=vectorstore,
            k=config["TOP_K"],
            n_clusters=config["N_CLUSTERS"]
        )

        cache.save_processed_chunks(
            processed_key,
            processed_chunks
        )

    cache.save_video_metadata(
        video_id,
        cache.vectorstore_path(vs_key),
        chunk_key,
        processed_key,
    )

    logger.info("Running agent")

    result = run_agent(
        task=task,
        query=user_query,
        chunks=chunks,
        processed_chunks=processed_chunks,
        vectorstore=vectorstore,
        llm=llm,
        return_meta=True,
        trace_id=trace_id
    )

    output = result["output"]

    agent_metadata = result["metadata"]

    agent_metadata["latency"] = (
        time.time() - start
    )

    fully_cached = (
        transcript_cache_hit
        and chunk_cache_hit
        and vs_cache_hit
        and processed_cache_hit
    )

    agent_metadata["strategy"] = STRATEGY_BY_MODE.get(mode, mode.upper())
    agent_metadata["chunk_count"] = len(chunks)
    agent_metadata["cached"] = fully_cached
    agent_metadata["cache"] = {
        "cached": fully_cached,
        "transcript_hit": transcript_cache_hit,
        "chunk_hit": chunk_cache_hit,
        "vectorstore_hit": vs_cache_hit,
        "processed_hit": processed_cache_hit,
    }

    logger.info("Pipeline completed")

    log_event({
        "trace_id": trace_id,
        "query": user_query,
        "task": agent_metadata["task"],
        "mode": agent_metadata["mode"],
        "latency": agent_metadata["latency"],
        "fallback_triggered": agent_metadata["fallback_triggered"],
        "cache": agent_metadata["cache"]
    })

    return {
        "output": output,
        "metadata": agent_metadata,
        "chunks": chunks,
        "processed_chunks": processed_chunks,
    }