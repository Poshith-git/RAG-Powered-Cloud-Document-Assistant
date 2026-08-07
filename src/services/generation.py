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

# Broadened beyond the original 3 words -- covers the common phrasing
# patterns for "give me the enumerable things" questions, not just a
# fixed keyword list. This still won't catch everything, which is why
# there's also a short-answer fallback below.
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

# If the LLM's answer is this short, it's likely truncated to a single
# bullet from a longer list rather than a genuinely complete answer.
_SHORT_ANSWER_WORD_THRESHOLD = 6


def extract_list(context):
    """
    Extract a numbered or bulleted list from the retrieved context,
    generically -- not tied to any specific document or heading.
    Returns None if fewer than 2 list-like lines are found, to avoid
    false positives on a single stray numbered sentence.
    """
    lines = [l.strip() for l in context.split("\n") if l.strip()]
    list_lines = [
        l for l in lines
        if _NUMBERED_PATTERN.match(l) or _BULLETED_PATTERN.match(l)
    ]

    if len(list_lines) < 2:
        return None

    return "\n".join(list_lines)


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


def confidence_label(score):
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
        extracted = extract_list(context)
        answer = extracted if extracted else generate_answer(context, query)
    else:
        answer = generate_answer(context, query)

        # Safety net: even if the query wording didn't trip the list
        # detector, a suspiciously short answer against a context that
        # clearly contains a list is a sign the LLM truncated to one
        # bullet instead of answering fully -- prefer the full list.
        if len(answer.split()) <= _SHORT_ANSWER_WORD_THRESHOLD:
            extracted = extract_list(context)
            if extracted:
                answer = extracted

    label = confidence_label(float(scores[0]))

    return answer, context, label