from pypdf import PdfReader

def load_pdf(file_path):
    """
    Loads a PDF file from a file path
    and extracts all text.
    Compatible with local and HF Docker environment.
    """
    try:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        return text

    except Exception as e:
        print(f"PDF Loading Error: {e}")
        return ""
