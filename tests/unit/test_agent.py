from agent import (decide_mode, classify_intent, answer_with_rag, run_agent)
from guardrails.context_guard import FALLBACK_RESPONSE


class MockResponse:

    def __init__(self, content):
        self.content = content


class MockLLM:

    def __init__(self, response="summary"):
        self.response = response

    def invoke(self, prompt):
        return MockResponse(self.response)


class MockDocument:

    def __init__(self, content):
        self.page_content = content


class MockVectorstore:

    def similarity_search(self, query, k=3):
        return []


class MockVectorstoreWithResults:

    def similarity_search(self, query, k=3):

        return [
            MockDocument("machine learning basics"),
            MockDocument("deep learning models")
        ]

def test_decide_mode_small():

    chunks = ["a", "b", "c"]

    result = decide_mode(chunks)

    assert result == "small"


def test_decide_mode_medium():

    chunks = list(range(10))

    result = decide_mode(chunks)

    assert result == "medium"


def test_decide_mode_long():

    chunks = list(range(25))

    result = decide_mode(chunks)

    assert result == "long"

def test_classify_intent_summary():

    llm = MockLLM("summary")

    result = classify_intent(
        "summarize this video",
        llm
    )

    assert result == "summary"


def test_classify_intent_rag():

    llm = MockLLM("rag")

    result = classify_intent(
        "what did the speaker say about AI?",
        llm
    )

    assert result == "rag"

def test_classify_intent_keypoints():

    llm = MockLLM("keypoints")

    result = classify_intent(
        "give keypoints",
        llm
    )

    assert result == "keypoints"


def test_classify_intent_qa_gen():

    llm = MockLLM("qa_gen")

    result = classify_intent(
        "generate questions",
        llm
    )

    assert result == "qa_gen"

def test_classify_intent_invalid():

    llm = MockLLM("random_task")

    metadata = {"test": True}

    result = classify_intent(
        "hello",
        llm,
        metadata
    )

    assert result is None

    assert metadata["failure_reason"] == "INVALID_INTENT"

    assert metadata["fallback_triggered"] is True

def test_rag_no_retrieval():

    vectorstore = MockVectorstore()

    llm = MockLLM()

    metadata = {
        "retrieval": {}
    }

    result = answer_with_rag(
        query="What is machine learning?",
        vectorstore=vectorstore,
        llm=llm,
        metadata=metadata
    )

    assert result == FALLBACK_RESPONSE

    assert metadata["fallback_triggered"] is True

    assert metadata["failure_reason"] == "NO_RETRIEVED_CHUNKS"

    assert metadata["retrieval"]["top_k"] == 0


def test_rag_low_relevance_fallback(monkeypatch):

    vectorstore = MockVectorstoreWithResults()

    llm = MockLLM()

    metadata = {
        "retrieval": {}
    }

    def fake_validate_context(
        query_emb,
        chunk_embs
    ):
        return False, True, 0.1

    monkeypatch.setattr(
        "agent.validate_context",
        fake_validate_context
    )

    result = answer_with_rag(
        query="What is machine learning?",
        vectorstore=vectorstore,
        llm=llm,
        metadata=metadata
    )

    assert result == FALLBACK_RESPONSE

    assert metadata["fallback_triggered"] is True

    assert metadata["failure_reason"] == "LOW_CONTEXT_RELEVANCE"


def test_rag_success(monkeypatch):

    vectorstore = MockVectorstoreWithResults()

    llm = MockLLM("This is the final answer")

    metadata = {
        "retrieval": {}
    }

    def fake_validate_context(
        query_emb,
        chunk_embs
    ):
        return True, False, 0.9

    monkeypatch.setattr(
        "agent.validate_context",
        fake_validate_context
    )

    result = answer_with_rag(
        query="What is machine learning?",
        vectorstore=vectorstore,
        llm=llm,
        metadata=metadata
    )

    assert result == "This is the final answer"

    assert metadata["retrieval"]["top_k"] == 2

    assert metadata["retrieval"]["score"] == 0.9

def test_run_agent_summary_routing(monkeypatch):

    def fake_summary(
        processed_chunks,
        llm,
        mode
    ):
        return (
        "This is a sufficiently long summary output."
    )

    monkeypatch.setattr(
        "agent.generate_summary",
        fake_summary
    )

    llm = MockLLM("summary")

    result = run_agent(
        query="summarize this video",
        chunks=["a", "b"],
        processed_chunks=["processed"],
        vectorstore=None,
        llm=llm,
        return_meta=True
    )

    assert result["output"] == "This is a sufficiently long summary output."

    assert result["metadata"]["task"] == "summary"


def test_run_agent_keypoints_routing(monkeypatch):

    def fake_keypoints(
        processed_chunks,
        llm,
        mode
    ):
        return "The is sufficiently long keypoints output"

    monkeypatch.setattr(
        "agent.generate_keypoints",
        fake_keypoints
    )

    llm = MockLLM("keypoints")

    result = run_agent(
        query="give keypoints",
        chunks=["a", "b"],
        processed_chunks=["processed"],
        vectorstore=None,
        llm=llm,
        return_meta=True
    )

    assert result["output"] == "The is sufficiently long keypoints output"

    assert result["metadata"]["task"] == "keypoints"


def test_run_agent_qa_routing(monkeypatch):

    def fake_questions(
        query,
        vectorstore,
        llm
    ):
        return "This is sufficiently long questions output"

    monkeypatch.setattr(
        "agent.generate_questions",
        fake_questions
    )

    llm = MockLLM("qa_gen")

    result = run_agent(
        query="generate questions",
        chunks=["a", "b"],
        processed_chunks=["processed"],
        vectorstore="fake_vs",
        llm=llm,
        return_meta=True
    )

    assert result["output"] == "This is sufficiently long questions output"

    assert result["metadata"]["task"] == "qa_gen"


def test_run_agent_rag_routing(monkeypatch):

    def fake_rag(
        query,
        vectorstore,
        llm,
        metadata
    ):
        return "This is sufficiently long rag output"

    monkeypatch.setattr(
        "agent.answer_with_rag",
        fake_rag
    )

    llm = MockLLM("rag")

    result = run_agent(
        query="what is ai?",
        chunks=["a", "b"],
        processed_chunks=["processed"],
        vectorstore="fake_vs",
        llm=llm,
        return_meta=True
    )

    assert result["output"] == "This is sufficiently long rag output"

    assert result["metadata"]["task"] == "rag"

def test_run_agent_invalid_input(monkeypatch):

    def fake_validate_query(query):
        return False, "Query too short"

    monkeypatch.setattr(
        "agent.validate_query",
        fake_validate_query
    )

    llm = MockLLM("summary")

    result = run_agent(
        query="bad query",
        chunks=["a"],
        processed_chunks=["processed"],
        vectorstore=None,
        llm=llm,
        return_meta=True
    )

    assert result["output"] == "Query too short"

    assert result["metadata"]["failure_reason"] == "INVALID_INPUT"

    assert result["metadata"]["fallback_triggered"] is True


def test_run_agent_invalid_intent(monkeypatch):

    def fake_classify_intent(
        query,
        llm,
        metadata
    ):
        return None

    monkeypatch.setattr(
        "agent.classify_intent",
        fake_classify_intent
    )

    llm = MockLLM("summary")

    result = run_agent(
        query="hello",
        chunks=["a"],
        processed_chunks=["processed"],
        vectorstore=None,
        llm=llm
    )

    assert result == FALLBACK_RESPONSE


def test_run_agent_invalid_output(monkeypatch):

    def fake_validate_query(query):
        return True, None

    monkeypatch.setattr(
        "agent.validate_query",
        fake_validate_query
    )

    def fake_classify_intent(
        query,
        llm,
        metadata
    ):
        return "summary"

    monkeypatch.setattr(
        "agent.classify_intent",
        fake_classify_intent
    )

    def fake_summary(
        processed_chunks,
        llm,
        mode
    ):
        return "summary output"

    monkeypatch.setattr(
        "agent.generate_summary",
        fake_summary
    )

    def fake_validate_output(output):
        return False

    monkeypatch.setattr(
        "agent.validate_output",
        fake_validate_output
    )

    llm = MockLLM("summary")

    result = run_agent(
        query="summarize",
        chunks=["a"],
        processed_chunks=["processed"],
        vectorstore=None,
        llm=llm,
        return_meta=True
    )

    assert result["output"] == FALLBACK_RESPONSE

    assert result["metadata"]["failure_reason"] == "OUTPUT_INVALID"

    assert result["metadata"]["fallback_triggered"] is True