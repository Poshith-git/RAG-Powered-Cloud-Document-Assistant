import streamlit as st
from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from utils.generator import generate_answer

# ----------------------------------------
# Page Configuration
# ----------------------------------------
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

# ----------------------------------------
# Sidebar
# ----------------------------------------
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


# ----------------------------------------
# File Upload
# ----------------------------------------
uploaded_file = st.file_uploader("Upload a PDF file (Max 5MB)", type=["pdf"])

if uploaded_file is not None:

    # Prevent large file crash (HF free tier safe)
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("Please upload a PDF smaller than 5MB.")
        st.stop()

    with st.spinner("Processing document..."):

        # Load text
        text = load_pdf(uploaded_file)

        if not text.strip():
            st.error("Could not extract text from PDF.")
            st.stop()

        # Chunk
        chunks = chunk_text(text)

        # Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Create FAISS index
        index = create_faiss_index(embeddings)

    st.success("Document processed successfully!")

    # ----------------------------------------
    # Question Input
    # ----------------------------------------
    query = st.text_input("Ask a question about the document:")

    if query:

        with st.spinner("Retrieving relevant context..."):

            # Query embedding
            query_embedding = generate_embeddings([query])[0]

            # Search
            results, scores = search_index(index, query_embedding, chunks)

            if len(results) == 0:
                st.error("No relevant context found.")
                st.stop()

            context = results[0]
            confidence = float(scores[0])

        # ----------------------------------------
        # Generate Answer
        # ----------------------------------------
        with st.spinner("Generating answer..."):
            answer = generate_answer(context, query)

        st.subheader("📌 Generated Answer")
        st.write(answer)

        st.caption(f"Retrieval Confidence Score: {round(confidence, 4)}")

        with st.expander("📄 View Retrieved Context"):
            st.write(context)
