import streamlit as st
from utils.loader import load_pdf
from utils.chunker import chunk_text

st.set_page_config(page_title="Cloud-Based RAG Document Assistant")

st.title("📄 Cloud-Based RAG Document Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.warning("⚠ Large file detected. Processing may take longer and use more memory.")

    text = load_pdf(uploaded_file)

    st.subheader("Extracted Text Preview:")
    st.write(text[:1000])  # show first 1000 characters

    chunks = chunk_text(text)

    st.subheader("Chunk Information")
    st.write(f"Total Chunks Created: {len(chunks)}")
