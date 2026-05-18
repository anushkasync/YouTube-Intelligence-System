import time
from data.transcripts import extract_video_id, get_transcript
from rag.chunker import chunk_text, Chunker
from rag.vectorstore import create_vectorstore
from rag.embeddings import embedding_model
from logger import get_logger, log_event
import uuid
from agent import run_agent, decide_mode


def run_pipeline(
    youtube_url,
    user_query,
    cache,
    llm,
    config,
    return_chunks=False
):
    trace_id = uuid.uuid4().hex
    logger = get_logger(trace_id)

    logger.info("Pipeline started")
    start = time.time()
    
    logger.info("Fetching transcript")
    video_id = extract_video_id(youtube_url)

# TRANSCRIPT (CACHE)

    transcript = cache.get_transcript(video_id)
    logger.info("Transcript cache hit")

    if not transcript or not transcript.get("text") or transcript.get("text").strip() == "":
        logger.info("Transcript cache miss/invalid → fetching fresh transcript")

        transcript = get_transcript(youtube_url)

# FINAL VALIDATION (VERY IMPORTANT)
        if not transcript or not transcript.get("text") or transcript.get("text").strip() == "":
            logger.error("Empty transcript → stopping pipeline")

            return {
            "output": "No usable transcript found for this video.",
            "metadata": {
                "failed_stage": "transcript",
                "reason": transcript.get("error", "empty_or_missing_text")
            }
        }

        cache.save_transcript(video_id, transcript)

    else:
        logger.info("Transcript cache hit (valid)")

    text = transcript.get("text", "")

# CHUNKING (CACHE)

    chunk_key = cache.make_chunk_key(
    video_id,
    config["CHUNK_SIZE"],
    config["OVERLAP"]
)

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


# VECTORSTORE (CACHE)
    
    vs_key = cache.make_vectorstore_key(
    video_id,
    embedding_model_name=config["EMBEDDING_MODEL_NAME"],
    chunk_size=config["CHUNK_SIZE"],
    overlap=config["OVERLAP"]
)
    logger.info("Loading vectorstore")
    vectorstore = cache.load_vectorstore(vs_key, embedding_model)
    vs_cache_hit = vectorstore is not None

    if vectorstore is None:
        logger.info("Creating vectorstore")
        vectorstore = create_vectorstore(
            chunks,
            video_id,
            embedding_model
        )
    
        cache.save_vectorstore(vs_key, vectorstore)


# PROCESSING (CACHE)

    mode = decide_mode(chunks)
    logger.info(f"Processing chunks in {mode} mode")
    processed_key = cache.make_processed_key(video_id, mode)

    processed_chunks = cache.get_processed_chunks(processed_key)

    if processed_chunks is None:
        chunker = Chunker(embedding_model)

        processed_chunks = chunker.get_processed_chunks(
            chunks,
            mode=mode,
            vectorstore=vectorstore,
            k=config["TOP_K"],
            n_clusters=config["N_CLUSTERS"]
        )

        cache.save_processed_chunks(processed_key, processed_chunks)

  
# AGENT (NO CACHE HERE)

    logger.info("Running agent")
    result = run_agent(
        query=user_query,
        chunks=chunks,
        processed_chunks=processed_chunks,
        vectorstore=vectorstore,
        llm=llm,
        return_meta=True,
        trace_id=trace_id
    )

    output = result["output"]
    metadata = result["metadata"]

    metadata["latency"] = time.time() - start

    metadata["cache"] = {
    "chunk_hit": chunk_cache_hit,
    "vectorstore_hit": vs_cache_hit
} 
    
    logger.info("Pipeline completed")

    log_event({
    "trace_id": trace_id,
    "query": user_query,
    "task": metadata["task"],
    "mode": metadata["mode"],
    "latency": metadata["latency"],
    "fallback_triggered": metadata["fallback_triggered"],
    "cache": metadata["cache"]
})

    return {
    "output": output,
    "metadata": metadata,
    "chunks" : chunks,
    "processed_chunks": processed_chunks,   
}
    