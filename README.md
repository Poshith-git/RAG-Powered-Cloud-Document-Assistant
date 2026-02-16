---
title: RAG Document Assistant
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Cloud-Based RAG Document Assistant

An end-to-end Retrieval-Augmented Generation (RAG) web application built using:

- Streamlit
- FAISS
- SentenceTransformers
- FLAN-T5 (local model)

Users can upload a PDF document and ask natural language questions.
The system retrieves relevant content and generates AI-powered answers grounded in the document.

Pipeline:
PDF → Cleaning → Chunking → Embeddings → FAISS → LLM → Answer + Confidence
