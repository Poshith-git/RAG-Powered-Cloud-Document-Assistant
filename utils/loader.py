from pypdf import PdfReader
import re


def load_pdf(file):
    """
    Extract and aggressively clean PDF text.
    """
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        cleaned_lines = []

        for line in text.split("\n"):
            line = line.strip()

            if not line:
                continue

            # Remove index heading
            if line.lower().startswith("index"):
                continue

            # Remove lines that are mostly dots (TOC format)
            if re.search(r"\.{5,}", line):
                continue

            # Remove lines that are mostly numbers
            if re.fullmatch(r"[0-9.\s]+", line):
                continue

            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        return cleaned_text

    except Exception as e:
        return f"Error reading PDF: {str(e)}"
