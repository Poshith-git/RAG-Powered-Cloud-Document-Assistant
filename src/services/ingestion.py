"""
Ingestion service: turns an uploaded file into text chunks.
Kept separate from retrieval/generation so it can be unit-tested
without loading any ML models.
"""

from utils.loader import load_pdf
from utils.chunker import chunk_text
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# Security hardening (added per the design doc's Chapter 6 checklist,
# not implemented until now): basic validation to avoid resource
# exhaustion and to reject files that don't look like real PDFs before
# they ever reach parsing/embedding.
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB
PDF_MAGIC_BYTES = b"%PDF-"


def _validate_upload(uploaded_file):
    """
    Validate file size and PDF magic bytes BEFORE attempting to parse.
    Checking magic bytes (not just the file extension) matters because
    a file can be renamed to .pdf without actually being one -- Streamlit's
    type=["pdf"] filter only checks the extension client-side.
    """
    uploaded_file.seek(0, 2)  # seek to end
    size = uploaded_file.tell()
    uploaded_file.seek(0)  # reset for the actual read

    if size == 0:
        raise ValueError("The uploaded file is empty.")

    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File is too large ({size / (1024 * 1024):.1f}MB). "
            f"Maximum allowed is {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f}MB."
        )

    header = uploaded_file.read(len(PDF_MAGIC_BYTES))
    uploaded_file.seek(0)  # reset again for the real parse

    if header != PDF_MAGIC_BYTES:
        raise ValueError(
            "This file does not appear to be a valid PDF (missing PDF header). "
            "It may be corrupted or renamed from a different file type."
        )


def ingest_pdf(uploaded_file):
    """
    Validate, load, and chunk an uploaded PDF.
    Raises ValueError for invalid uploads (size, type) or empty extracted text.
    """
    _validate_upload(uploaded_file)

    text = load_pdf(uploaded_file)

    if not text.strip():
        raise ValueError("No readable text found in the PDF.")

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    return chunks