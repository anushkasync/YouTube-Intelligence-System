import os

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
SMALL_LIMIT = 5
MEDIUM_LIMIT = 20
N_CLUSTERS = 3

LLM_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "qwen/qwen3-32b"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# Whisper Fallback Settings
WHISPER_ENABLED = True  
WHISPER_MODEL = "base"  
WHISPER_LANGUAGE = "en"  

ENV = os.getenv("ENV", "prod").lower()
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Persistent cache root — separate from logs.
# Railway: mount volume at /data (override with CACHE_DIR env var).
_DEFAULT_CACHE_DIR = (
    "/data/cache"
    if os.path.isdir("/data")
    else os.path.join(BASE_DIR, "data", "cache")
)
CACHE_DIR = os.getenv("CACHE_DIR", _DEFAULT_CACHE_DIR)

SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "systems.log")
BENCHMARK_RESULTS_FILE = os.path.join(
    LOG_DIR,
    "benchmark_results.json"
)

CONFIG = {
    "CHUNK_SIZE": CHUNK_SIZE,
    "OVERLAP": OVERLAP,
    "TOP_K": TOP_K,
    "SMALL_LIMIT": SMALL_LIMIT,
    "MEDIUM_LIMIT": MEDIUM_LIMIT,
    "N_CLUSTERS": N_CLUSTERS,
    "LLM_MODEL": LLM_MODEL,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
    "WHISPER_ENABLED": WHISPER_ENABLED,
    "WHISPER_MODEL": WHISPER_MODEL,
    "WHISPER_LANGUAGE": WHISPER_LANGUAGE,
    "ENV": ENV,
    "CACHE_DIR": CACHE_DIR,
    "SYSTEM_LOG_FILE": SYSTEM_LOG_FILE,
    "BENCHMARK_RESULTS_FILE": BENCHMARK_RESULTS_FILE
}