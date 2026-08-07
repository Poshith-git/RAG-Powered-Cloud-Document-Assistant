"""
Thin Streamlit UI. All actual logic lives in src/services/ so it can
be unit-tested and, later, reused behind a FastAPI layer without
rewriting it.

Run with: streamlit run src/ui/streamlit_app.py
"""
import sys
import os

# Ensure the project root is on sys.path so `src.services...` imports
# resolve correctly no matter what directory Streamlit is launched from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import streamlit as st

from src.services.ingestion import ingest_pdf
from src.services.retrieval import build_index, retrieve
from src.services.generation import answer_question


st.set_page_config(page_title="Cloud-Based RAG Document Assistant", layout="wide")

st.title("📄 Cloud-Based RAG Document Assistant")

st.markdown(
    """
Upload a PDF document and ask questions about it.
The system retrieves relevant sections using semantic search
and generates AI-powered answers grounded strictly in the document.
"""
)

with st.sidebar:
    st.header("🔎 About This Project")
    st.markdown(
        """
**Architecture:**
Document → Chunking → E5 Embeddings → FAISS → Hybrid Answering

**Models Used:**
- Generator: FLAN-T5-base
- Embeddings: intfloat/e5-base-v2
- Vector Store: FAISS (Cosine Similarity)
- Hybrid: General-purpose list extraction + LLM generation
"""
    )
    st.divider()
    st.caption("v3.0 — restructured into src/services/, bugfixed")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        with st.spinner("Processing document..."):
            chunks = ingest_pdf(uploaded_file)
            index, _ = build_index(chunks)
        st.success("Document processed successfully!")
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.stop()

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Ask a question about the document:")
    with col2:
        if st.button("Clear"):
            st.rerun()

    if query:
        try:
            with st.spinner("Retrieving and generating answer..."):
                results, scores = retrieve(index, chunks, query)
                answer, context, confidence = answer_question(results, scores, query)

            st.success("Answer generated successfully!")

            st.subheader("📌 Generated Answer")
            st.write(answer)
            st.caption(f"Retrieval Confidence: {confidence} ({round(float(scores[0]), 3)})")

            with st.expander("📄 View Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")