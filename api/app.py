import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from logger import get_logger
from config.config import CONFIG

from utils.cache_manager import CacheManager
from utils.llm import GROQLLM

from run_pipeline import run_pipeline
from evaluations.full_suite import run_full_suite

from api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
)

load_dotenv()

ENV = os.getenv("ENV", "prod").lower()

logger = get_logger("FASTAPI")

cache = None
llm = None

def require_dev():

    if ENV != "dev":

        raise HTTPException(
            status_code=403,
            detail="Developer-only endpoint"
        )
    
@asynccontextmanager
async def lifespan(app: FastAPI):

    global cache
    global llm

    logger.info("FastAPI application starting")

    cache = CacheManager()

    logger.info("Cache manager initialized")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY missing")
        raise ValueError("GROQ_API_KEY missing")

    llm = GROQLLM(
        api_key=api_key,
        model=CONFIG["LLM_MODEL"],
        cache_manager=cache
    )

    logger.info("LLM initialized")

    yield

    logger.info("FastAPI application shutting down")

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
    description="Production API for Agentic RAG YouTube System",
    lifespan=lifespan
)

@app.get("/")
def root():

    return {
        "message": "Agentic RAG API Running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    return {
        "status": "healthy",
        "cache": "initialized" if cache else "not_initialized",
        "llm": "initialized" if llm else "not_initialized"
    }


@app.post(
    "/query",
    response_model=QueryResponse
)
def query_video(request: QueryRequest):

    trace_id = str(uuid.uuid4())

    logger.info(
        f"[{trace_id}] Query endpoint called"
    )

    start = time.time()

    try:

        result = run_pipeline(
            youtube_url=request.youtube_url,
            user_query=request.query,
            cache=cache,
            llm=llm,
            config=CONFIG
        )

        output = result.get("output", "")
        metadata = result.get("metadata", {})

        #metadata["trace_id"] = trace_id
        metadata["latency"] = round(
            time.time() - start,
            2
        )

        logger.info(
            f"[{trace_id}] Query completed successfully"
        )

        return {
            "success": True,
            "output": output,
            "metadata": metadata
        }

    except Exception as e:

        logger.exception(
            f"[{trace_id}] Query failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "Internal server error",
                "trace_id": trace_id
            }
        )

@app.post("/evaluation")
def evaluation():

    require_dev()

    trace_id = str(uuid.uuid4())

    logger.info(
        f"[{trace_id}] Evaluation endpoint called"
    )

    start = time.time()

    try:

        result = run_full_suite(
            cache=cache,
            llm=llm,
            config=CONFIG,
            mode="eval"
        )

        latency = round(
            time.time() - start,
            2
        )

        logger.info(
            f"[{trace_id}] Evaluation completed"
        )

        return {
            "success": True,
            "mode": result["mode"],
            "evaluations": result["evaluations"],
            "latency": latency,
            "trace_id": trace_id
        }

    except Exception as e:
        logger.error(f"[{trace_id}] Evaluation failed: {str(e)}")

        return {
            "success": False,
            "error": "Evaluation service temporarily unavailable. Please retry.",
            "trace_id": trace_id
        }

@app.post("/benchmark")
def benchmark():

    require_dev()
    trace_id = str(uuid.uuid4())

    logger.info(
        f"[{trace_id}] Benchmark endpoint called"
    )

    start = time.time()

    try:

        result = run_full_suite(
            cache=cache,
            llm=llm,
            config=CONFIG,
            mode="benchmark"
        )

        latency = round(
            time.time() - start,
            2
        )

        logger.info(
            f"[{trace_id}] Benchmark completed"
        )

        return {
            "success": True,
            "mode": result["mode"],
            "benchmark": result["benchmark"],
            "latency": latency,
            "trace_id": trace_id
        }

    except Exception as e:

        logger.exception(
            f"[{trace_id}] Benchmark failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "trace_id": trace_id
            }
        )