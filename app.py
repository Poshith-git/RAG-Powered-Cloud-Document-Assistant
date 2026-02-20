import streamlit as st
from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from utils.generator import generate_answer

@st.cache_data
def cached_embeddings(chunks):
    return generate_embeddings(chunks)

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
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("🔎 About This Project")

    st.markdown("""
**Architecture:**
Document → Chunking → Embeddings → FAISS → LLM

**Models Used:**
- Generator: FLAN-T5-small  
- Embeddings: all-MiniLM-L6-v2  
- Vector Store: FAISS (Cosine Similarity)
""")

    st.divider()
    st.caption("Version 1.0 – Stable Deployment")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:

    try:
        with st.spinner("Processing document..."):
            text = load_pdf(uploaded_file)

            if not text.strip():
                st.error("No readable text found in the PDF.")
                st.stop()

            chunks = chunk_text(text)
            embeddings = cached_embeddings(chunks)
            index = create_faiss_index(embeddings)

        st.success("Document processed successfully!")

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.stop()

    # -------------------------------
    # Question Section
    # -------------------------------
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input("Ask a question about the document:")

    with col2:
        if st.button("Clear"):
            st.experimental_rerun()

    if query:

        try:
            with st.spinner("Retrieving and generating answer..."):

                # Embed query
                query_embedding = generate_embeddings([query])[0]

                # Retrieve top 3 chunks
                results, scores = search_index(index, query_embedding, chunks)

                top_k = 3
                selected_chunks = results[:top_k]

                # Combine chunks
                context = "\n\n".join(selected_chunks)

                # Limit context size for small model stability
                context = context[:1500]

                # Average confidence score
                confidence = float(sum(scores[:top_k]) / top_k)

                # Generate answer
                answer = generate_answer(context, query)

            st.success("Answer generated successfully!")

            # -------------------------------
            # Display Answer
            # -------------------------------
            st.subheader("📌 Generated Answer")
            st.write(answer)

            st.caption(f"Retrieval Confidence Score: {round(confidence, 4)}")

            with st.expander("📄 View Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
