import streamlit as st
import re
from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from utils.generator import generate_answer


# -------------------------------------------------
# General-Purpose Numbered/Bulleted List Extraction
# -------------------------------------------------
def extract_numbered_list(context):
    """
    Extract a numbered or bulleted list from the retrieved context,
    generically -- not tied to any specific document or heading.
    Looks for the densest run of list-like lines in the context and
    returns that run, so it works on any document that happens to
    contain a list near the retrieved passage.
    """

    lines = [l.strip() for l in context.split("\n") if l.strip()]

    numbered_pattern = re.compile(r"^\d+[\.\)]\s+.+")
    bulleted_pattern = re.compile(r"^[-*•]\s+.+")

    list_lines = [
        l for l in lines
        if numbered_pattern.match(l) or bulleted_pattern.match(l)
    ]

    # Require at least 2 list-like lines to avoid false positives
    # on a single stray numbered sentence.
    if len(list_lines) < 2:
        return None

    return "\n".join(list_lines)


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
    st.caption("Version 2.9 – Hybrid RAG, Bugfixed")


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
            st.rerun()

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
                    # Do NOT trim context for list extraction --
                    # truncating could cut a list off mid-way.
                    extracted = extract_numbered_list(context)
                    if extracted:
                        answer = extracted
                    else:
                        # Falls through to the LLM; use the same
                        # single trim point as the generation path below.
                        answer = generate_answer(context, query)
                else:
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