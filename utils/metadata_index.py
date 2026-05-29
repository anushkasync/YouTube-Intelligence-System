import os
import sqlite3

from logger import get_logger

logger = get_logger("METADATA_INDEX")


class MetadataIndex:

    def __init__(self, db_path):

        self.db_path = db_path

        db_dir = os.path.dirname(db_path)

        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

        logger.info(f"Metadata index initialized at {db_path}")

    def _connect(self):

        return sqlite3.connect(self.db_path)

    def _init_db(self):

        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_cache (
                    video_id TEXT PRIMARY KEY,
                    faiss_vectorstore_path TEXT NOT NULL,
                    chunk_key TEXT NOT NULL,
                    processed_key TEXT NOT NULL,
                    updated_at REAL DEFAULT (unixepoch('now'))
                )
            """)

    def get(self, video_id):

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT faiss_vectorstore_path, chunk_key, processed_key
                FROM video_cache
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "faiss_vectorstore_path": row[0],
            "chunk_key": row[1],
            "processed_key": row[2],
        }

    def upsert(
        self,
        video_id,
        faiss_vectorstore_path,
        chunk_key,
        processed_key,
    ):

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO video_cache (
                    video_id,
                    faiss_vectorstore_path,
                    chunk_key,
                    processed_key,
                    updated_at
                )
                VALUES (?, ?, ?, ?, unixepoch('now'))
                ON CONFLICT(video_id) DO UPDATE SET
                    faiss_vectorstore_path = excluded.faiss_vectorstore_path,
                    chunk_key = excluded.chunk_key,
                    processed_key = excluded.processed_key,
                    updated_at = excluded.updated_at
                """,
                (
                    video_id,
                    faiss_vectorstore_path,
                    chunk_key,
                    processed_key,
                ),
            )

        logger.info(f"Metadata index updated for video_id={video_id}")

    def delete(self, video_id):

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM video_cache WHERE video_id = ?",
                (video_id,),
            )
