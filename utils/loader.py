from pypdf import PdfReader
import io

def load_pdf(uploaded_file):
    """
    Load PDF directly from Streamlit uploaded file
    WITHOUT saving to disk (HF-safe).
    """

    try:
        # Read file bytes
        pdf_bytes = uploaded_file.read()

        # Convert to in-memory binary stream
        pdf_stream = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_stream)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:
        raise Exception(f"PDF loading failed: {str(e)}")
