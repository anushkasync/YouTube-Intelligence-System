import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

MODES = {
    "Summary": {
        "task": "summary",
        "query": "Summarize this video",
    },
    "Key Points": {
        "task": "keypoints",
        "query": "List the key points from this video",
    },
    "Generate Questions": {
        "task": "qa_gen",
        "query": "Generate study questions from this video",
    },
    "Ask a Question": {
        "task": "rag_qa",
        "query": None,
    },
}

st.set_page_config(
    page_title="YouTube Intelligence",
    page_icon="▶",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metrics-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        color: #334155;
        margin: 1rem 0;
    }
    .output-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">YouTube Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Analyze any YouTube video — summaries, key points, questions, and Q&amp;A.</p>',
    unsafe_allow_html=True,
)


def call_query_api(youtube_url: str, query: str, task: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={
            "youtube_url": youtube_url,
            "query": query,
            "task": task,
        },
        timeout=300,
    )

    if response.status_code != 200:
        detail = response.text
        try:
            payload = response.json()
            detail = payload.get("detail", detail)
            if isinstance(detail, dict):
                detail = detail.get("error", str(detail))
        except ValueError:
            pass
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    return response.json()


def build_metrics_bar(metadata: dict) -> str:
    cached = metadata.get("cached", False)
    strategy = metadata.get("strategy", metadata.get("mode", "—")).upper()
    chunk_count = metadata.get("chunk_count", "—")
    latency = metadata.get("latency", 0)

    status = "⚡ Cached" if cached else "⚙️ Fresh"
    return (
        f"{status} | Strategy: {strategy} | "
        f"Chunks: {chunk_count} | Time: {latency}s"
    )


with st.form("query_form", clear_on_submit=False):
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    mode = st.selectbox(
        "Mode",
        options=list(MODES.keys()),
    )

    user_question = None
    if mode == "Ask a Question":
        user_question = st.text_area(
            "Your question",
            placeholder="What is the main argument of this video?",
            height=100,
        )

    submitted = st.form_submit_button("Process", use_container_width=True)

if submitted:
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL.")
    elif mode == "Ask a Question" and (not user_question or len(user_question.strip()) < 3):
        st.error("Please enter a question (at least 3 characters).")
    else:
        mode_config = MODES[mode]
        query = user_question.strip() if mode == "Ask a Question" else mode_config["query"]

        with st.spinner("Processing video…"):
            try:
                result = call_query_api(
                    youtube_url=youtube_url.strip(),
                    query=query,
                    task=mode_config["task"],
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to the API at `{API_BASE_URL}`. "
                    "Make sure the FastAPI server is running."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("The request timed out. Try again with a shorter video.")
                st.stop()
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        if not result.get("success"):
            st.error("The API returned an unsuccessful response.")
            st.stop()

        metadata = result.get("metadata", {})
        output = result.get("output", "")

        st.markdown("---")
        st.subheader("Results")

        st.markdown(
            f'<div class="metrics-bar">{build_metrics_bar(metadata)}</div>',
            unsafe_allow_html=True,
        )

        if metadata.get("cached"):
            st.success("⚡ Cached — returned instantly, zero processing cost")
        else:
            st.success("⚙️ Processed fresh")

        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.markdown(output)
        st.markdown("</div>", unsafe_allow_html=True)

        if metadata.get("fallback_triggered"):
            reason = metadata.get("failure_reason") or "Unknown"
            st.warning(f"Fallback triggered: {reason}")

with st.expander("How it works"):
    st.markdown(
        """
        The backend automatically selects a retrieval strategy based on video length:

        - **Short videos** : **Raw retrieval** — all chunks used directly
        - **Medium videos** : **Top-K semantic retrieval** — most relevant chunks selected
        - **Long videos** : **K-Means clustering retrieval** — representative chunks from clusters

        Cached videos skip re-processing and return results instantly.
        """
    )
