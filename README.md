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

## Architecture Design
This system is a deterministic pipeline for YouTube video understanding, optimized for low latency, cost efficiency, and scalable retrieval-based reasoning.

### Transcript Extraction
* Primary: YouTube Transcript API
* Fallback: Whisper transcription

Ensures fast-path retrieval with graceful degradation when transcripts are unavailable.

### Adaptive Chunking Strategy
| Video Type | Processing Strategy | Reason
|---|---|---|
| Small Videos | Raw chunks | Full context retained without additional processing
| Medium Videos | Top-K retrieval | Improves relevance filtering
| Long Videos | KMeans clustering | Improves coverage by selecting diverse semantic centroids and reducing redundancy

Different video lengths introduce varying challenges in retrieval and token efficiency.

### Single-Pass LLM Generation
All processed chunks are merged and passed through a single LLM call per module.
This design replaces multi-stage summarization pipelines with a single inference step, significantly reducing latency and cost while preserving contextual completeness.

### Mode-Specific Prompting
Prompts are dynamically selected based on video length:
* Short → detail preservation
* Medium → relevance-focused generation
* Long → abstraction and coverage optimization

Prompt versions are maintained for iterative improvement and reproducibility.

### Multi-Layer Caching
Caching is applied across:
* transcripts
* chunks
* processed chunks
* vector store
* LLM outputs

Reduces redundant computation and API overhead across repeated requests.

### Evaluation & Observability
Includes:
* unit, integration, and smoke tests
* structured logging and tracing
* LLM-based evaluation (clarity, completeness, usefulness)
* RAG evaluation (faithfulness, relevance)

### Throttling & Rate Control
Implements request throttling to ensure system stability under multi-stage LLM pipelines and external API rate limits.

### Model Strategy
A single primary LLM is used across modules.
Multi-model comparison is deprioritized due to:
* multiple LLM calls per request (classification + generation + evaluation)
* increased latency and cost
* external provider rate limits (especially in free-tier environments)

### Environment Separation
* Development: benchmarking, evaluation endpoints, health checks
* Production: inference-only API surface

### Deterministic Orchestration vs Autonomous Agents
The system uses a deterministic pipeline instead of autonomous agents because the workflow is fixed and well-structured (transcript extraction, chunking, retrieval, generation, and evaluation). 

Agent-based orchestration was avoided to reduce unnecessary complexity and overhead.
