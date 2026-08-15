import io
import sys
import types

# Stub heavy ML deps -- ingestion.py imports utils.loader which imports
# pdfplumber (fine, lightweight) but chunker.py has no heavy deps either,
# so this module is actually testable without stubbing much. Kept the
# pattern consistent with other test files regardless.
for mod in ["streamlit"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

from src.services.ingestion import _validate_upload, MAX_FILE_SIZE_BYTES


def test_validate_upload_rejects_empty_file():
    f = io.BytesIO(b"")
    try:
        _validate_upload(f)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_validate_upload_rejects_oversized_file():
    f = io.BytesIO(b"%PDF-1.4\n" + b"0" * (MAX_FILE_SIZE_BYTES + 1))
    try:
        _validate_upload(f)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "too large" in str(e).lower()


def test_validate_upload_rejects_non_pdf_content():
    # A file renamed to .pdf but that isn't actually one -- e.g. a
    # plain text file. Magic-byte check should catch this even though
    # a naive extension-only check would not.
    f = io.BytesIO(b"This is just plain text, not a PDF.")
    try:
        _validate_upload(f)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "pdf" in str(e).lower()


def test_validate_upload_accepts_valid_pdf_header_and_resets_stream_position():
    f = io.BytesIO(b"%PDF-1.4\n%valid pdf content here")
    _validate_upload(f)  # should not raise
    # Stream position must be reset to 0 so the real parser can read from the start.
    assert f.tell() == 0