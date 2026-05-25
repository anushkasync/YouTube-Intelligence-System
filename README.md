# YouTube RAG Intelligence System

A modular, agent-driven Retrieval-Augmented Generation (RAG) system that processes YouTube videos and enables intelligent downstream tasks such as summarization, key-point extraction, question generation, and contextual question answering.

The system is designed with a strong focus on **caching, cost-aware LLM usage, and scalable pipeline orchestration**, making it suitable for real-world LLM application constraints such as rate limits and latency control.

## Description
This project ingests YouTube videos, extracts transcripts, processes them into structured representations, and runs multiple NLP pipelines on top of the content. It supports both **generative tasks (summarization, Q&A)** and **analytical tasks (key points, question generation)** using a shared cached processing layer.

The system is built to be **LLM-cost aware and production-oriented**, ensuring minimal redundant API usage while maintaining modularity across different AI tasks.

## Core Modules
### - Summarization Module
### - Key-Points Generation Module
### - Question Generation Module
### - RAG Answering Module
