from pypdf import PdfReader

def load_pdf(file):
    """
    Extract text from uploaded PDF file with error handling.
    """
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:
        return f"Error reading PDF: {str(e)}"

