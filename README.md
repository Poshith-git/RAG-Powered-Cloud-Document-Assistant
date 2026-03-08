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

A **Retrieval-Augmented Generation (RAG)** web application that allows users to upload PDF documents and ask questions based on the document content. The system retrieves relevant sections using semantic search and generates grounded answers using a language model.

---

## Architecture

User Question  
↓  
Query Embedding (E5 Model)  
↓  
FAISS Vector Search  
↓  
Relevant Document Chunks  
↓  
FLAN-T5 Answer Generation  
↓  
Answer + Confidence Score

---

## Features

- Upload and query **PDF documents**
- Semantic search using **E5 embeddings**
- Fast vector retrieval with **FAISS**
- Answer generation using **FLAN-T5**
- Retrieval **confidence scoring**
- Deployable using **Streamlit and Docker**

---

## Tech Stack

Python  
Streamlit  
FAISS  
Sentence Transformers  
Hugging Face Transformers  
Docker