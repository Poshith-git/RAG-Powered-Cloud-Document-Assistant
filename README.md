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

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![RAG](https://img.shields.io/badge/AI-RAG_System-purple)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![HuggingFace](https://img.shields.io/badge/LLM-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A cloud-deployed **Retrieval-Augmented Generation (RAG)** application that enables users to upload PDF documents and ask questions grounded in the document content.

The system performs **semantic retrieval using vector embeddings** and generates answers using a **language model**, reducing hallucination and improving response relevance.

---

## Architecture

User Question  
↓  
Query Embedding (E5 Model)  
↓  
FAISS Vector Similarity Search  
↓  
Top-K Relevant Document Chunks  
↓  
FLAN-T5 Context-Aware Answer Generation  
↓  
Answer + Retrieval Confidence Score

---

## Features

- Upload and query **PDF documents**
- **Semantic document retrieval** using E5 embeddings
- Efficient vector similarity search using **FAISS**
- Context-aware answer generation using **FLAN-T5**
- Retrieval **confidence scoring**
- Deployable as a **Streamlit web application**

---

## Tech Stack

Python  
Streamlit  
FAISS  
Sentence Transformers  
Hugging Face Transformers  
Docker

---

## Demo

Try the live application on Hugging Face Spaces:

https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

---

## Example Use Cases

Upload a PDF and ask questions such as:

- What is the Spiral Model?
- Explain the advantages of the Spiral Model.
- What are the key stages of the Spiral Model?

---

## License

MIT License