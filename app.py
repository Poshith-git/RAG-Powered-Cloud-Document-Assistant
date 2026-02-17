import streamlit as st
from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from utils.generator import generate_answer

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Cloud-Based RAG Document Assistant",
    layout="wide"
)

st.title("📄 Cloud-Based RAG Document Assistant")

st.markdown("""
Upload a PDF document and ask questions about it.  
The system retrieves relevant sections using semantic search  
and generates AI-powered answers grounded in the document.
""")

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.header("🔎 About This Project")
    st.write("""
This application uses Retrieval-Augmented Generation (RAG).

Pipeline:
Document → Chunking → Embeddings → FAISS Retrieval → LLM Generation
""")
    st.write("Model: FLAN-T5-small")
    st.write("Embedding: all-MiniLM-L6-v2")
    st.write("Vector Store: FAISS (Cosine Similarity)")

# ---------------------------------
# File Upload (HF SAFE)
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file is not None:

    try:
        # Reset pointer (important for HF Spaces)
        uploaded_file.seek(0)

        # Load text from PDF (memory-based only)
        text = load_pdf(uploaded_file)

        if not text or not text.strip():
            st.error("No readable text found in the PDF.")
            st.stop()

        # ---------------------------------
        # Chunking
        # ---------------------------------
        chunks = chunk_text(text)

        if not chunks:
            st.error("Text chunking failed.")
            st.stop()

        # ---------------------------------
        # Embeddings
        # ---------------------------------
        embeddings = generate_embeddings(chunks)

        # ---------------------------------
        # Create FAISS Index
        # ---------------------------------
        index = create_faiss_index(embeddings)

        st.success("✅ Document processed successfully!")

        # ---------------------------------
        # Question Section
        # ---------------------------------
        query = st.text_input("Ask a question about the document:")

        if query:

            query_embedding = generate_embeddings([query])[0]

            results, scores = search_index(index, query_embedding, chunks)

            if not results:
                st.warning("No relevant context found.")
                st.stop()

            context = results[0]
            confidence = float(scores[0])

            with st.spinner("Generating answer..."):
                answer = generate_answer(context, query)

            st.success("Answer generated successfully!")

            # Display Answer
            st.subheader("📌 Generated Answer")
            st.write(answer if answer.strip() else "No answer generated.")

            # Confidence
            st.caption(f"Retrieval Confidence Score: {round(confidence, 4)}")

            # Context Viewer
            with st.expander("📄 View Retrieved Context"):
                st.write(context)

    except Exception as e:
        st.error("An unexpected error occurred.")
        st.error(str(e))
