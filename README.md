# YouTube RAG Intelligence System

An intelligent RAG system for processing YouTube videos and performing downstream AI tasks such as summarization, key-point extraction, question generation, and contextual question answering.

The system extracts and processes video transcripts into structured representations while using shared caching and modular pipelines to reduce redundant LLM calls, optimize cost, and improve scalability.

## Core Modules
- Summarization Module
- Key-Points Generation Module
- Question Generation Module
- RAG Answering Module

## Architecture Diagram
<img width="600" height="600" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/5da46ac7-379a-443c-8c9b-5a04c22fc60a" />

## Installation

### 1. Clone repository

```bash
git clone <repo-url>
cd project
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run server

```bash
python -m uvicorn api.app:app --reload
```

## Tech Stack

- Python
- FastAPI
- LangChain
- Groq APIs
- FAISS
- Youtube transcript API
- Whisper
- Uvicorn
- Pytest

## Architecture Decisions

### Transcript Extraction
* Primary: YouTube Transcript API
* Fallback: Whisper transcription

### Adaptive Chunking Strategy
| Video Type | Processing Strategy | Reason
|---|---|---|
| Small Videos | Raw chunks | Full context retained without additional processing
| Medium Videos | Top-K retrieval | Improves relevance filtering
| Long Videos | KMeans clustering | Improves coverage by selecting diverse semantic centroids and reducing redundancy

### Single-Pass LLM Generation
All processed chunks are merged and passed through a single LLM call per module.
This design replaces multi-stage summarization pipelines with a single inference step, reducing latency and API usage while preserving context.

### Mode-Specific Prompting
Prompts are dynamically selected based on video length:
* Short → detail preservation
* Medium → relevance-focused generation
* Long → abstraction and coverage optimization

### Multi-Layer Caching
Caching is applied across:
* transcripts
* raw and processed chunks
* vector store
* LLM outputs

### Evaluation & Observability
Includes:
* unit, integration, and smoke tests
* structured logging and tracing
* LLM-based and RAG evaluation

### Throttling & Rate Control
Implements request throttling to ensure system stability under multi-stage LLM pipelines and external API rate limits.

### Model Strategy
A single primary LLM is used across modules.
Multi-model comparison is deprioritized due to:
* multiple LLM calls per request (classification + generation)
* external provider rate limits (especially in free-tier environments)

### Environment Separation
* Development: benchmarking, evaluation endpoints, health checks
* Production: inference-only API surface

### Deterministic Orchestration vs Autonomous Agents
The system uses a deterministic pipeline instead of autonomous agents because the workflow is fixed and well-structured (transcript extraction, chunking, retrieval, generation, and evaluation). 

Agent-based orchestration was avoided to reduce unnecessary complexity and overhead.
