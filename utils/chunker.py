import re

_NUMBERED_PATTERN = re.compile(r"^\d+[\.\)]\s+.+")
_BULLETED_PATTERN = re.compile(r"^[-*\u2022]\s+.+")


def _is_list_line(line):
    return bool(_NUMBERED_PATTERN.match(line) or _BULLETED_PATTERN.match(line))


def _group_into_units(paragraphs):
    """
    Group consecutive list-like lines (bulleted/numbered) into a single
    atomic unit, so a chunk boundary can never fall in the middle of a
    list. Non-list paragraphs remain their own units.

    Without this, a long list of bullets (e.g. a 6-item "Responsibilities"
    section) can get split across two separate chunks. Since retrieval
    only pulls back a limited number of chunks, the second half of the
    list can silently be missing from the answer even though it's in
    the document.
    """
    units = []
    current_list_run = []

    for para in paragraphs:
        if _is_list_line(para):
            current_list_run.append(para)
        else:
            if current_list_run:
                units.append("\n".join(current_list_run))
                current_list_run = []
            units.append(para)

    if current_list_run:
        units.append("\n".join(current_list_run))

    return units


def chunk_text(text, chunk_size=800, overlap=150):
    """
    Paragraph-aware chunking to preserve section boundaries, with a
    trailing overlap carried into the next chunk so that context isn't
    lost at chunk boundaries. Bulleted/numbered lists are grouped into
    atomic units first (see _group_into_units) so a list is never split
    across two chunks, even if that means a chunk exceeds chunk_size.
    """

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    units = _group_into_units(paragraphs)

    chunks = []
    current_chunk = ""

    for unit in units:
        if len(current_chunk) + len(unit) < chunk_size:
            current_chunk += " " + unit
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # carry the trailing `overlap` characters into the next chunk
                tail = current_chunk[-overlap:] if overlap > 0 else ""
                current_chunk = tail + " " + unit
            else:
                # current_chunk is empty (e.g. a single unit, like a long
                # list, is already >= chunk_size) -- keep the whole unit
                # together rather than splitting it.
                current_chunk = unit

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks