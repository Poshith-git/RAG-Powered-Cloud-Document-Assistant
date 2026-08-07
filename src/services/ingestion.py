"""
Ingestion service: turns an uploaded file into text chunks.
Kept separate from retrieval/generation so it can be unit-tested
without loading any ML models.
"""

from utils.loader import load_pdf
from utils.chunker import chunk_text
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest_pdf(uploaded_file):
    """
    Load a PDF and split it into overlapping chunks.
    Raises ValueError if no readable text is found.
    """
    text = load_pdf(uploaded_file)

    if not text.strip():
        raise ValueError("No readable text found in the PDF.")

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    return chunks