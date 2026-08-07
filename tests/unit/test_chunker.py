from utils.chunker import chunk_text


def test_chunk_text_returns_nonempty_list():
    text = "Paragraph one.\nParagraph two.\nParagraph three."
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 0


def test_chunk_text_respects_overlap():
    # Force multiple chunks with a small chunk_size, then confirm
    # consecutive chunks share trailing/leading text (the overlap bug fix).
    text = "\n".join([f"This is paragraph number {i} with some content." for i in range(10)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # The start of chunk[1] should contain the tail of chunk[0]
    tail_of_first = chunks[0][-20:].strip()
    assert any(word in chunks[1] for word in tail_of_first.split() if len(word) > 3)


def test_chunk_text_empty_input():
    assert chunk_text("", chunk_size=100, overlap=10) == []