"""
Generation service: builds context from retrieved chunks and produces
an answer, using either the general-purpose list extractor or the
FLAN-T5 generator.
"""

import re
from utils.generator import generate_answer
from src.config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
)

_LIST_TRIGGER_WORDS = (
    "advantages", "disadvantages", "list", "qualifications",
    "requirements", "responsibilities", "benefits", "perks",
    "features", "steps", "types", "skills",
)
_ENUMERATION_PATTERN = re.compile(
    r"\bwhat are\b|\bwhich\b.*\bare\b|\blist\b|\bname\b.*\ball\b", re.IGNORECASE
)

_NUMBERED_PATTERN = re.compile(r"^\d+[\.\)]\s+.+")
_BULLETED_PATTERN = re.compile(r"^[-*\u2022]\s+.+")

_SHORT_ANSWER_WORD_THRESHOLD = 6

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "does", "do", "did", "this", "that", "for", "of", "to", "in", "on",
    "and", "or", "with", "required", "role", "job", "most",
}

# A word (or word FAMILY -- see _words_match) present in more than this
# fraction of chunks is treated as generic to the document. Lowered from
# 0.5 based on diagnostic evidence: on the O*NET test document, the
# document's own subject word "developers" only word-family-matched in
# 2 of 6 chunks (33%) and "software" matched in exactly 3 of 6 (50%,
# sitting right on the old boundary) -- both survived the old threshold
# as "rare/discriminating" despite being the single most generic
# concept in the whole document, which defeated the out-of-scope
# refusal check. This fraction is an approximation, not a precise
# general solution -- a word's frequency doesn't repeat evenly across
# every section even when it IS the document's core subject, and the
# right fraction likely depends on document size/structure. A more
# principled fix (e.g. explicitly excluding the document's own
# title/subject words) is noted as future work rather than tuned
# further here.
_GENERIC_WORD_CHUNK_FRACTION = 0.3


def _words_match(word_a, word_b):
    """
    Prefix-CONTAINMENT match (shorter word is a genuine prefix of the
    longer one) -- e.g. "wage"/"wages", "annual"/"annually",
    "develop"/"developers" all match. Deliberately NOT fixed-length
    truncation: that approach was found (via the eval harness) to cause
    false-positive collisions between unrelated words that happen to
    share the same first few characters, e.g. "companies" and
    "computers" both truncate to "comp" but share no real relationship.
    """
    shorter, longer = (word_a, word_b) if len(word_a) <= len(word_b) else (word_b, word_a)
    return longer.startswith(shorter)


def _word_overlap_count(words_a, words_b):
    """Count words in words_a that are a prefix-match with any word in words_b."""
    return sum(1 for wa in words_a if any(_words_match(wa, wb) for wb in words_b))


def _has_word_overlap(words_a, words_b):
    return _word_overlap_count(words_a, words_b) > 0


def _significant_words(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def _chunk_word_sets(chunks):
    return [_significant_words(c) for c in chunks]


def _prefix_match_chunk_count(word, chunk_word_sets):
    """
    Count chunks where ANY word prefix-matches `word` -- this clusters
    word families together (e.g. "develop"/"developers"/"developing"
    all count toward the same concept) rather than counting each exact
    token in isolation. Found via the eval harness: exact-token
    frequency counting missed that "developers" (query) is really the
    same generic concept as "develop" (appearing in the Overview
    section), so it wasn't recognized as generic to a document that is
    entirely about software developers -- letting it through as a
    "rare/discriminating" word and defeating the out-of-scope refusal
    check for an unrelated question that happened to mention
    "developers".
    """
    return sum(1 for words in chunk_word_sets if any(_words_match(word, w) for w in words))


def _rare_words(words, chunk_word_sets):
    """
    Filter a word set down to words that are NOT generic to this
    document (see _prefix_match_chunk_count). Falls back to the
    unfiltered set if filtering would eliminate everything, since a
    very small candidate chunk set otherwise degenerates (every word
    trivially looks "generic" when there's only one chunk to check
    against).
    """
    total_chunks = len(chunk_word_sets)
    if total_chunks == 0:
        return words
    threshold = _GENERIC_WORD_CHUNK_FRACTION * total_chunks
    filtered = {w for w in words if _prefix_match_chunk_count(w, chunk_word_sets) <= threshold}
    return filtered if filtered else words


def _extract_list_from_text(text):
    """
    Extract a numbered or bulleted list from a single block of text.
    Returns None if fewer than 2 list-like lines are found, to avoid
    false positives on a single stray numbered sentence.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    list_lines = [
        l for l in lines
        if _NUMBERED_PATTERN.match(l) or _BULLETED_PATTERN.match(l)
    ]

    if len(list_lines) < 2:
        return None

    return "\n".join(list_lines)


def extract_list(ordered_chunks, query=None):
    """
    Extract a list from the retrieved chunks, section-aware: each chunk
    is checked independently for a qualifying list (>= 2 list-like
    lines), and the best-matching chunk's list is returned.

    "Best matching" isn't simply "first in retrieval-ranked order": the
    top-ranked chunk by cosine similarity is not always the chunk whose
    SECTION actually matches the query. When a query is given, each
    candidate chunk is scored by how many RARE (document-discriminating)
    query words appear in that chunk, using word-family prefix matching.
    Ties fall back to retrieval-ranked order.
    """
    candidates = []
    for rank, chunk in enumerate(ordered_chunks):
        extracted = _extract_list_from_text(chunk)
        if extracted:
            candidates.append((chunk, extracted, rank))

    if not candidates:
        return None

    if query is None:
        return candidates[0][1]

    chunk_word_sets = _chunk_word_sets(ordered_chunks)
    query_words = _rare_words(_significant_words(query), chunk_word_sets)

    def relevance_score(candidate):
        chunk, extracted, rank = candidate
        chunk_words = _rare_words(_significant_words(chunk), chunk_word_sets)
        overlap = _word_overlap_count(query_words, chunk_words)
        # Higher overlap wins; ties broken by earlier retrieval rank
        # (negative so lower rank/higher relevance sorts first).
        return (overlap, -rank)

    best = max(candidates, key=relevance_score)
    return best[1]


def _looks_like_list_question(query_lower):
    if any(word in query_lower for word in _LIST_TRIGGER_WORDS):
        return True
    return bool(_ENUMERATION_PATTERN.search(query_lower))


def order_by_intent(chunks, query_lower):
    """Boost chunks that look like definitions when the query asks 'what is X'."""
    definition_priority, other_chunks = [], []

    wants_definition = query_lower.startswith("what is") or query_lower.startswith("define")

    for chunk in chunks:
        chunk_lower = chunk.lower()
        if wants_definition and (" is a " in chunk_lower or " is an " in chunk_lower):
            definition_priority.append(chunk)
        else:
            other_chunks.append(chunk)

    return definition_priority + other_chunks


def build_context(ordered_chunks):
    return "\n\n".join(chunk.strip() for chunk in ordered_chunks)


def _has_rare_word_overlap(query, ordered_chunks):
    """
    Cosine similarity from dense embeddings rarely drops very low even
    for genuinely irrelevant chunks, since FAISS always returns a
    "top-k closest" match regardless of true relevance. As a cheap
    additional signal, require that at least one RARE (document-
    discriminating) query word overlaps with the top retrieved chunk --
    excluding words/word-families that are generic to the whole
    document, which was found to make a plain overlap check pass
    trivially for almost any query on a narrowly-scoped document.
    """
    if not ordered_chunks:
        return True  # nothing to check against; don't penalize

    chunk_word_sets = _chunk_word_sets(ordered_chunks)
    query_words = _rare_words(_significant_words(query), chunk_word_sets)

    if not query_words:
        return True  # no discriminating words to check; don't penalize

    top_chunk_words = _significant_words(ordered_chunks[0])
    return _has_word_overlap(query_words, top_chunk_words)


def confidence_label(score, query=None, ordered_chunks=None):
    if query is not None and ordered_chunks is not None and not _has_rare_word_overlap(query, ordered_chunks):
        return "Low"

    if score > CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    elif score > CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def answer_question(chunks, scores, query):
    """
    Full generation pipeline: order chunks, build context, decide
    between list extraction and LLM generation, and compute confidence.
    Returns (answer, context, confidence_label).
    """
    query_lower = query.lower()

    ordered = order_by_intent(chunks, query_lower)
    context = build_context(ordered)

    if _looks_like_list_question(query_lower):
        extracted = extract_list(ordered, query=query)
        answer = extracted if extracted else generate_answer(context, query)
    else:
        answer = generate_answer(context, query)

        looks_like_truncated_list_item = bool(
            _NUMBERED_PATTERN.match(answer.strip()) or _BULLETED_PATTERN.match(answer.strip())
        )
        if looks_like_truncated_list_item and len(answer.split()) <= _SHORT_ANSWER_WORD_THRESHOLD:
            extracted = extract_list(ordered, query=query)
            if extracted:
                answer = extracted

    label = confidence_label(float(scores[0]), query=query, ordered_chunks=ordered)

    return answer, context, label