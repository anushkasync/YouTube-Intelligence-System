# YouTube RAG Intelligence System

A modular, agent-driven Retrieval-Augmented Generation (RAG) system that processes YouTube videos and enables intelligent downstream tasks such as summarization, key-point extraction, question generation, and contextual question answering.

The system is designed with a strong focus on **caching, cost-aware LLM usage, and scalable pipeline orchestration**, making it suitable for real-world LLM application constraints such as rate limits and latency control.

## Description
This project ingests YouTube videos, extracts transcripts, processes them into structured representations, and runs multiple NLP pipelines on top of the content. It supports both **generative tasks (summarization, Q&A)** and **analytical tasks (key points, question generation)** using a shared cached processing layer.

The system is built to be **LLM-cost aware and production-oriented**, ensuring minimal redundant API usage while maintaining modularity across different AI tasks.

## Core Modules
- Summarization Module
- Key-Points Generation Module
- Question Generation Module
- RAG Answering Module

## Architecture Overview
This system is a deterministic pipeline for YouTube video understanding, optimized for low latency, cost efficiency, and scalable retrieval-based reasoning.

### Transcript Extraction
* Primary: YouTube Transcript API
* Fallback: Whisper transcription
Ensures fast-path retrieval with graceful degradation when transcripts are unavailable.

### Adaptive Chunking Strategy
Video Type	Strategy
Short	Raw chunks
Medium	Top-K retrieval
Long	K-Means clustering
Different video lengths introduce varying challenges in retrieval and token efficiency.
* Short videos: full context retained without additional processing
* Medium videos: Top-K retrieval improves relevance filtering
* Long videos: K-Means improves coverage by selecting diverse semantic centroids and reducing redundancy

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
Ensures safe deployment and reduced attack surface.

### Deterministic Orchestration vs Autonomous Agents
The system intentionally uses a deterministic pipeline architecture instead of autonomous agent-based workflows.
The problem space is highly structured and consists of fixed stages
- transcript extraction
- chunking and preprocessing
- retrieval or clustering
- single-pass generation
- evaluation and benchmarking
Autonomous agent frameworks were avoided as they introduce unnecessary orchestration complexity and multi-step reasoning overhead for a well-defined pipeline problem.
