# api/app.py

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from logger import get_logger
from config.config import CONFIG

from utils.cache_manager import CacheManager
from utils.llm import OpenRouterLLM

from run_pipeline import run_pipeline
from evaluations.full_suite import run_full_suite

from api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
)

# =====================================================
# ENV
# =====================================================

load_dotenv()

ENV = os.getenv("ENV", "prod").lower()

logger = get_logger("FASTAPI")

# =====================================================
# GLOBAL SINGLETONS
# =====================================================

cache = None
llm = None

# =====================================================
# STARTUP / SHUTDOWN
# =====================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    global cache
    global llm

    logger.info("FastAPI application starting")

    # ---------------------------
    # CACHE INIT
    # ---------------------------

    cache = CacheManager(
        base_dir=CONFIG["CACHE_DIR"]
    )

    logger.info("Cache manager initialized")

    # ---------------------------
    # LLM INIT
    # ---------------------------

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        logger.error("OPENROUTER_API_KEY missing")
        raise ValueError("OPENROUTER_API_KEY missing")

    llm = OpenRouterLLM(
        api_key=api_key,
        model=CONFIG["LLM_MODEL"],
        cache_manager=cache
    )

    logger.info("LLM initialized")

    yield

    logger.info("FastAPI application shutting down")


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
    description="Production API for Agentic RAG YouTube System",
    lifespan=lifespan
)

# =====================================================
# ROOT
# =====================================================


@app.get("/")
def root():

    return {
        "message": "Agentic RAG API Running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# =====================================================
# HEALTH ENDPOINT
# =====================================================


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


# =====================================================
# QUERY ENDPOINT
# =====================================================


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

        # =============================================
        # RUN ORIGINAL PIPELINE
        # =============================================

        result = run_pipeline(
            youtube_url=request.youtube_url,
            user_query=request.query,
            cache=cache,
            llm=llm,
            config=CONFIG
        )

        output = result.get("output", "")
        metadata = result.get("metadata", {})

        # =============================================
        # ADD EXTRA API METADATA
        # =============================================

        metadata["trace_id"] = trace_id
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
                "error": str(e),
                "trace_id": trace_id
            }
        )


# =====================================================
# EVALUATION ENDPOINT
# =====================================================


@app.post("/evaluation")
def evaluation():

    trace_id = str(uuid.uuid4())

    logger.info(
        f"[{trace_id}] Evaluation endpoint called"
    )

    start = time.time()

    try:

        # =============================================
        # SAME AS:
        # python main.py --eval
        # =============================================

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

# =====================================================
# BENCHMARK ENDPOINT
# =====================================================


@app.post("/benchmark")
def benchmark():

    trace_id = str(uuid.uuid4())

    logger.info(
        f"[{trace_id}] Benchmark endpoint called"
    )

    start = time.time()

    try:

        # =============================================
        # SAME AS:
        # python main.py --benchmark
        # =============================================

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