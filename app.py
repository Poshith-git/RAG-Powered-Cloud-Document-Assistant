import streamlit as st
import numpy as np
from utils.generator import generate_answer


from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings, model
from utils.retriever import create_faiss_index, search_index

st.set_page_config(page_title="Cloud-Based RAG Document Assistant")

st.title("📄 Cloud-Based RAG Document Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Extract text
    text = load_pdf(uploaded_file)

    if not text or text.startswith("Error"):
        st.error("Error extracting text from PDF.")
    else:
        # Chunk text
        chunks = chunk_text(text)

        st.write(f"Total Chunks Created: {len(chunks)}")

        if "index" not in st.session_state:
            embeddings = generate_embeddings(chunks)
            st.session_state.index = create_faiss_index(embeddings)
            st.session_state.chunks = chunks

        index = st.session_state.index
        chunks = st.session_state.chunks


        # Query input
        query = st.text_input("Ask a question about the document")

        if query:
            query_embedding = model.encode(query)
            results = search_index(index, query_embedding, chunks)

            # Combine retrieved chunks into single context
            context = results[0]   # Only use most relevant chunk

            st.subheader("Generated Answer:")

            answer = generate_answer(context, query)

            st.write(answer)

