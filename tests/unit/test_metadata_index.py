import os

import pytest

from utils.cache_manager import CacheManager
from utils.metadata_index import MetadataIndex


@pytest.fixture
def metadata_index(tmp_path):

    db_path = tmp_path / "metadata.db"
    return MetadataIndex(str(db_path))


def test_metadata_index_upsert_and_get(metadata_index):

    metadata_index.upsert(
        "vid123",
        "/cache/vectorstores/abc123",
        "chunk_key_abc",
        "processed_key_xyz",
    )

    result = metadata_index.get("vid123")

    assert result == {
        "faiss_vectorstore_path": "/cache/vectorstores/abc123",
        "chunk_key": "chunk_key_abc",
        "processed_key": "processed_key_xyz",
    }


def test_metadata_index_miss(metadata_index):

    assert metadata_index.get("nonexistent") is None


def test_metadata_index_upsert_overwrites(metadata_index):

    metadata_index.upsert(
        "vid123",
        "/path/v1",
        "chunk_v1",
        "processed_v1",
    )

    metadata_index.upsert(
        "vid123",
        "/path/v2",
        "chunk_v2",
        "processed_v2",
    )

    result = metadata_index.get("vid123")

    assert result["faiss_vectorstore_path"] == "/path/v2"
    assert result["chunk_key"] == "chunk_v2"
    assert result["processed_key"] == "processed_v2"


def test_metadata_index_delete(metadata_index):

    metadata_index.upsert(
        "vid123",
        "/path/v1",
        "chunk_v1",
        "processed_v1",
    )

    metadata_index.delete("vid123")

    assert metadata_index.get("vid123") is None


def test_cache_manager_metadata_persistence(tmp_path):

    cache1 = CacheManager(base_dir=tmp_path)

    cache1.save_video_metadata(
        "vid123",
        os.path.join(tmp_path, "vectorstores", "abc"),
        "chunk_key_abc",
        "processed_key_xyz",
    )

    cache2 = CacheManager(base_dir=tmp_path)

    result = cache2.lookup_video_metadata("vid123")

    assert result == {
        "faiss_vectorstore_path": os.path.join(tmp_path, "vectorstores", "abc"),
        "chunk_key": "chunk_key_abc",
        "processed_key": "processed_key_xyz",
    }
