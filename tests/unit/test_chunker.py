# test_chunker.py

import pytest

from rag.chunker import clean_text, chunk_text

def test_clean_text_newlines():
    text = "hello\n\nworld\tthis  is   test"
    result = clean_text(text)
    assert result == "hello world this is test"

def test_clean_text_strip():
    text = "   hello world   "
    result = clean_text(text)
    assert result == "hello world"

def test_clean_text_empty():
    text = ""
    result = clean_text(text)
    assert result == ""


def test_small_text_single_chunk():

    text = "hello world this is a short transcript"

    chunks = chunk_text(
        text,
        CHUNK_SIZE=100,
        OVERLAP=20
    )

    assert len(chunks) == 1

def test_chunk_size_limit():

    text = "word " * 50

    chunks = chunk_text(
        text,
        CHUNK_SIZE=10,
        OVERLAP=2
    )

    for chunk in chunks:
        assert len(chunk.split()) <= 10

def test_chunk_overlap():

    text = " ".join([f"word{i}" for i in range(30)])

    chunks = chunk_text(
        text,
        CHUNK_SIZE=10,
        OVERLAP=2
    )

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    overlap_words = first_chunk_words[-2:]

    assert overlap_words == second_chunk_words[:2]

def test_no_empty_chunks():

    text = "word " * 40

    chunks = chunk_text(
        text,
        CHUNK_SIZE=10,
        OVERLAP=2
    )

    assert all(chunk.strip() != "" for chunk in chunks)

def test_chunk_size_limit():

    text = "word " * 50

    chunks = chunk_text(
        text,
        CHUNK_SIZE=10,
        OVERLAP=5
    )

    for chunk in chunks:
        assert len(chunk.split()) <= 10


def test_invalid_overlap():

    with pytest.raises(ValueError):

        chunk_text(
            "word " * 20,
            CHUNK_SIZE=10,
            OVERLAP=10
        )