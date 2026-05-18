# utils/cache_manager.py
import os
import json
import hashlib
from logger import get_logger
from langchain_community.vectorstores import FAISS

logger = get_logger("CACHE_MANAGER")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def stable_hash(value):
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True)

    return hashlib.sha256(str(value).encode()).hexdigest()


# =========================================================
# CACHE MANAGER
# =========================================================

class CacheManager:

    def __init__(self, base_dir="cache"):

        self.base_dir = base_dir

        ensure_dir(base_dir)

        # json files
        self.transcript_path = os.path.join(base_dir, "transcripts.json")
        self.chunk_path = os.path.join(base_dir, "chunks.json")
        self.processed_chunk_path = os.path.join(base_dir, "processed_chunks.json")
        self.llm_path = os.path.join(base_dir, "llm_outputs.json")

        # vectorstores
        self.vectorstore_dir = os.path.join(base_dir, "vectorstores")
        ensure_dir(self.vectorstore_dir)

        # load memory
        self.transcripts = self._load_json(self.transcript_path)
        self.chunks = self._load_json(self.chunk_path)
        self.processed_chunks = self._load_json(self.processed_chunk_path)
        self.llm_outputs = self._load_json(self.llm_path)

    # =====================================================
    # INTERNAL
    # =====================================================

    def _load_json(self, path):
        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception:
            logger.error(f"Caching failed: unable to load file {path}")
            return {}

    def _save_json(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception:
            logger.error("Caching failed: unable to save data")

    # =====================================================
    # TRANSCRIPTS
    # =====================================================

    def get_transcript(self, video_id):
        return self.transcripts.get(video_id)

    def save_transcript(self, video_id, transcript):

        self.transcripts[video_id] = transcript

        self._save_json(
            self.transcript_path,
            self.transcripts
        )

    # =====================================================
    # CHUNKS
    # =====================================================

    def make_chunk_key(self, video_id, chunk_size, overlap):

        return stable_hash({
            "video_id": video_id,
            "chunk_size": chunk_size,
            "overlap": overlap
        })

    def get_chunks(self, key):
        return self.chunks.get(key)

    def save_chunks(self, key, chunks):

        self.chunks[key] = chunks

        self._save_json(
            self.chunk_path,
            self.chunks
        )

    # =====================================================
    # PROCESSED CHUNKS
    # =====================================================

    def make_processed_key(self, video_id, mode):

        return stable_hash({
            "video_id": video_id,
            "mode": mode
        })

    def get_processed_chunks(self, key):
        return self.processed_chunks.get(key)

    def save_processed_chunks(self, key, processed):

        self.processed_chunks[key] = processed

        self._save_json(
            self.processed_chunk_path,
            self.processed_chunks
        )

    # =====================================================
    # VECTORSTORE
    # =====================================================
    def make_vectorstore_key(self,video_id, embedding_model_name, chunk_size, overlap):
        return stable_hash({
        "video_id": video_id,
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "overlap": overlap
    })

    def vectorstore_path(self, key):

        return os.path.join(
            self.vectorstore_dir,
            key
        )

    def save_vectorstore(self, key, vectorstore):
        try:
            path = self.vectorstore_path(key)
            ensure_dir(path)
            vectorstore.save_local(path)

        except Exception:
            logger.error("Caching failed: vectorstore save failed")

    def load_vectorstore(self, key, embedding_model):
        path = self.vectorstore_path(key)

        if not os.path.exists(path):
            return None

        try:
            return FAISS.load_local(
                path,
                embedding_model,
                allow_dangerous_deserialization=True
            )

        except Exception:
            logger.error("Caching failed: vectorstore load failed")
            return None

    # =====================================================
    # LLM OUTPUTS
    # =====================================================

    def make_llm_key(self, prompt, model):

        return stable_hash({
            "prompt": prompt,
            "model": model
        })

    def get_llm_output(self, key):
        return self.llm_outputs.get(key)

    def save_llm_output(self, key, output):
        try:
            self.llm_outputs[key] = output
            self._save_json(self.llm_path, self.llm_outputs)

        except Exception as e:
            logger.error(f"Failed to save LLM output: {str(e)}")

