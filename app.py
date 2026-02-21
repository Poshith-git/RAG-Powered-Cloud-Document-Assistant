import streamlit as st
import re
from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from utils.generator import generate_answer


# -------------------------------------------------
# Section-Bounded List Extraction Utility
# -------------------------------------------------
def extract_numbered_list(context):
    """
    Extract numbered list items only from the
    'Advantages of the Spiral Model' section.
    """

    # Focus only on Advantages section
    if "Advantages of the Spiral Model" in context:
        section = context.split("Advantages of the Spiral Model", 1)[1]
    else:
        section = context

    # Stop extraction before unrelated headings
    stop_keywords = [
        "Disadvantages",
        "When To Use",
        "When a project",
        "Example"
    ]

    for keyword in stop_keywords:
        if keyword in section:
            section = section.split(keyword, 1)[0]

    pattern = r"\d+\.\s.*?(?=\s\d+\.|\Z)"
    matches = re.findall(pattern, section, re.DOTALL)

    if not matches:
        return None

    cleaned = [item.strip() for item in matches]
    return "\n\n".join(cleaned)


# -------------------------------------------------
# Cached Embeddings
# -------------------------------------------------
@st.cache_data
def cached_embeddings(chunks):
    return generate_embeddings(chunks, is_query=False)


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Cloud-Based RAG Document Assistant",
    layout="wide"
)

st.title("📄 Cloud-Based RAG Document Assistant")

st.markdown(
    """
Upload a PDF document and ask questions about it.
The system retrieves relevant sections using semantic search
and generates AI-powered answers grounded strictly in the document.
"""
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("🔎 About This Project")

    st.markdown("""
**Architecture:**  
Document → Chunking → E5 Embeddings → FAISS → Hybrid Answering  

**Models Used:**  
- Generator: FLAN-T5-base  
- Embeddings: intfloat/e5-base-v2  
- Vector Store: FAISS (Cosine Similarity)  
- Hybrid: Rule-based list extraction
""")

    st.divider()
    st.caption("Version 2.8 – Final Hybrid RAG")


# -------------------------------------------------
# File Upload
# -------------------------------------------------
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

    # -------------------------------------------------
    # Question Section
    # -------------------------------------------------
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input("Ask a question about the document:")

    with col2:
        if st.button("Clear"):
            st.experimental_rerun()

    if query:

        try:
            with st.spinner("Retrieving and generating answer..."):

                query_lower = query.lower()

                # Query embedding (E5 format)
                query_embedding = generate_embeddings(
                    [query],
                    is_query=True
                )[0]

                # Retrieve top 8 chunks
                results, scores = search_index(
                    index,
                    query_embedding,
                    chunks,
                    top_k=8
                )

                # Intent-aware ordering (definition boost)
                definition_priority = []
                other_chunks = []

                for chunk in results:
                    chunk_lower = chunk.lower()

                    if query_lower.startswith("what is") or query_lower.startswith("define"):
                        if " is a " in chunk_lower or " is an " in chunk_lower:
                            definition_priority.append(chunk)
                        else:
                            other_chunks.append(chunk)
                    else:
                        other_chunks.append(chunk)

                final_order = definition_priority + other_chunks

                # Merge context
                context = "\n\n".join(chunk.strip() for chunk in final_order)

                # -------------------------------------------------
                # Hybrid Answer Logic
                # -------------------------------------------------
                if (
                    "advantages" in query_lower
                    or "disadvantages" in query_lower
                    or "list" in query_lower
                ):
                    # DO NOT trim context for list extraction
                    extracted = extract_numbered_list(context)
                    if extracted:
                        answer = extracted
                    else:
                        answer = generate_answer(context[:1500], query)
                else:
                    # Trim context only for LLM generation
                    context = context[:1500]
                    answer = generate_answer(context, query)

                # Confidence score (Top-1 similarity)
                confidence = float(scores[0])

                if confidence > 0.80:
                    confidence_label = "High"
                elif confidence > 0.65:
                    confidence_label = "Medium"
                else:
                    confidence_label = "Low"

            st.success("Answer generated successfully!")

            # -------------------------------------------------
            # Display Answer
            # -------------------------------------------------
            st.subheader("📌 Generated Answer")
            st.write(answer)

            st.caption(
                f"Retrieval Confidence: {confidence_label} "
                f"({round(confidence, 3)})"
            )

            with st.expander("📄 View Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")