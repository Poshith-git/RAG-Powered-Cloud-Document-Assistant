"""
Integration test for the pdfplumber-based PDF loader, run against the
real sample PDF already in the repo (public domain).
"""
from pathlib import Path
from utils.loader import load_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent.parent / "data" / "sample_pdfs" / "software_developers_onet_summary.pdf"


class _FakeUploadedFile:
    """Minimal shim matching Streamlit's UploadedFile.read() interface."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self._bytes = f.read()

    def read(self):
        return self._bytes


def test_load_pdf_extracts_nonempty_text():
    text = load_pdf(_FakeUploadedFile(SAMPLE_PDF))
    assert len(text) > 100


def test_load_pdf_preserves_numbered_list_items_on_separate_lines():
    # Regression test for the loader swap (pypdf -> pdfplumber): the
    # whole reason for switching was that pypdf's extraction could
    # collapse a document's structure into one run-on paragraph with no
    # newlines, breaking line-anchored list detection downstream.
    # pdfplumber's layout-aware extraction should keep each numbered
    # Tasks item on its own line.
    text = load_pdf(_FakeUploadedFile(SAMPLE_PDF))
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    numbered_lines = [l for l in lines if l[:2] in ("1.", "2.", "3.", "4.", "5.")]
    # At least the first few Tasks items should each be their own line.
    assert len(numbered_lines) >= 3