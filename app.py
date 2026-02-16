import streamlit as st
import tempfile
import os

from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from utils.generator import generate_answer

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Cloud-Based RAG Document Assistant",
    layout="wide"
)

st.title("📄 Cloud-Based RAG Document Assistant")

st.markdown(
    """
Upload a PDF document and ask questions about it.
The system retrieves relevant sections using semantic search
and generates AI-powered answers grounded in the document.
"""
)

# -------------------------------
# Sidebar Info
# -------------------------------
with st.sidebar:
    st.header("🔎 About This Project")
    st.write(
        """
This application uses Retrieval-Augmented Generation (RAG).

Pipeline:
Document → Chunking → Embeddings → FAISS Retrieval → LLM Generation
"""
    )
    st.write("Model: FLAN-T5-small")
    st.write("Embedding: all-MiniLM-L6-v2")
    st.write("Vector Store: FAISS (Cosine Similarity)")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:

    try:
        # Save file safely in HF-supported temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF text
        text = load_pdf(tmp_path)

        # Remove temp file after reading
        os.remove(tmp_path)

        if not text.strip():
            st.error("❌ Could not extract text from PDF.")
            st.stop()

        # Chunk text
        chunks = chunk_text(text)

        # Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Create FAISS index
        index = create_faiss_index(embeddings)

        st.success("✅ Document processed successfully!")

        # -------------------------------
        # Question Input
        # -------------------------------
        query = st.text_input("Ask a question about the document:")

        if query:

            # Generate embedding for query
            query_embedding = generate_embeddings([query])[0]

            # Retrieve relevant chunks
            results, scores = search_index(index, query_embedding, chunks)

            if not results:
                st.warning("No relevant context found.")
                st.stop()

            # Use best chunk only (better for small model)
            context = results[0]
            confidence = float(scores[0])

            # -------------------------------
            # Generate Answer
            # -------------------------------
            with st.spinner("Generating answer..."):
                answer = generate_answer(context, query)

            if not answer.strip():
                st.warning("Model returned empty response.")
            else:
                st.subheader("📌 Generated Answer")
                st.write(answer)

            # Confidence Score
            st.caption(f"Retrieval Confidence Score: {round(confidence, 4)}")

            # Expandable Retrieved Context
            with st.expander("📄 View Retrieved Context"):
                st.write(context)

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")
