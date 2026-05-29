import os
import json
import hashlib
from logger import get_logger
from langchain_community.vectorstores import FAISS
from config.config import CACHE_DIR
from utils.metadata_index import MetadataIndex

logger = get_logger("CACHE_MANAGER")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def stable_hash(value):
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True)

    return hashlib.sha256(str(value).encode()).hexdigest()

class CacheManager:

    def __init__(self, base_dir=None):

        self.base_dir = base_dir or CACHE_DIR

        logger.info(
            f"Initializing cache manager at {self.base_dir}"
        )
        ensure_dir(self.base_dir)

        self.transcript_path = os.path.join(self.base_dir, "transcripts.json")
        self.chunk_path = os.path.join(self.base_dir, "chunks.json")
        self.processed_chunk_path = os.path.join(self.base_dir, "processed_chunks.json")
        self.llm_path = os.path.join(self.base_dir, "llm_outputs.json")

        self.vectorstore_dir = os.path.join(self.base_dir, "vectorstores")
        ensure_dir(self.vectorstore_dir)

        self.metadata_db_path = os.path.join(self.base_dir, "metadata.db")
        self.metadata_index = MetadataIndex(self.metadata_db_path)

        self.transcripts = self._load_json(self.transcript_path)
        self.chunks = self._load_json(self.chunk_path)
        self.processed_chunks = self._load_json(self.processed_chunk_path)
        self.llm_outputs = self._load_json(self.llm_path)


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

    # TRANSCRIPTS

    def get_transcript(self, video_id):
        return self.transcripts.get(video_id)

    def save_transcript(self, video_id, transcript):

        self.transcripts[video_id] = transcript

        self._save_json(
            self.transcript_path,
            self.transcripts
        )

    # CHUNKS

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

    # PROCESSED CHUNKS

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

    # VECTORSTORE

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

        except Exception as e:
            logger.error(
        f"Vectorstore save failed: {str(e)}"
    )
    def load_vectorstore(self, key, embedding_model):
        return self.load_vectorstore_from_path(
            self.vectorstore_path(key),
            embedding_model,
        )

    def load_vectorstore_from_path(self, path, embedding_model):

        logger.info(f"Checking vectorstore path: {path}")

        if not os.path.exists(path):
            logger.warning(f"Vectorstore cache miss: {path}")
            return None

        try:
            vectorstore = FAISS.load_local(
                path,
                embedding_model,
                allow_dangerous_deserialization=True,
            )

            logger.info(f"Vectorstore loaded successfully: {path}")
            return vectorstore

        except Exception as e:
            logger.error(f"Vectorstore load failed: {str(e)}")
            return None

    # METADATA INDEX (SQLite — pointers only, no embeddings/chunks)

    def lookup_video_metadata(self, video_id):

        return self.metadata_index.get(video_id)

    def save_video_metadata(
        self,
        video_id,
        faiss_vectorstore_path,
        chunk_key,
        processed_key,
    ):

        self.metadata_index.upsert(
            video_id,
            faiss_vectorstore_path,
            chunk_key,
            processed_key,
        )

    # LLM OUTPUTS

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

