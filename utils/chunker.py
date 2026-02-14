def chunk_text(text, chunk_size=600, overlap=150):
    """
    Split text into overlapping chunks and remove TOC-like chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Skip chunks that are too short
        if len(chunk.strip()) > 200:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
