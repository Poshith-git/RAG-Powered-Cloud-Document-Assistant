def chunk_text(text, chunk_size=800, overlap=150):
    """
    Paragraph-aware chunking to preserve section boundaries,
    with a trailing overlap carried into the next chunk so that
    context isn't lost at chunk boundaries.
    """

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            # carry the trailing `overlap` characters into the next chunk
            tail = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = tail + " " + para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks