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


def test_chunk_text_keeps_long_list_intact():
    # Regression test: a 6-item bulleted list was previously getting
    # split across two chunks when chunk_size was small relative to
    # the list length, silently dropping the second half of the list
    # from any single chunk's context. This was found by testing the
    # deployed app against a real job-posting PDF ("List all the
    # responsibilities...") where only 3 of 6 bullets came back.
    text = "\n".join([
        "Responsibilities",
        "• Design, build, and evaluate scalable, reliable systems.",
        "• Come up with efficient algorithms for performance and cost.",
        "• Use different tools and frameworks to produce insights.",
        "• Build effective automation frameworks for continuous feedback.",
        "• Own projects end-to-end from scoping to delivery.",
        "• Work with a fun team building exciting applications.",
        "Qualifications",
        "• Strong CS fundamentals.",
    ])

    # Deliberately small chunk_size -- small enough that the list would
    # have been split across chunks under the old paragraph-only logic.
    chunks = chunk_text(text, chunk_size=150, overlap=20)

    # Every bullet from the Responsibilities list must appear together
    # in the SAME chunk.
    list_chunk = next(c for c in chunks if "Design, build, and evaluate" in c)
    assert "Come up with efficient algorithms" in list_chunk
    assert "Use different tools and frameworks" in list_chunk
    assert "Build effective automation frameworks" in list_chunk
    assert "Own projects end-to-end" in list_chunk
    assert "Work with a fun team" in list_chunk