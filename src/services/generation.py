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

        # Safety net: only kick in if the LLM's own answer LOOKS like a
        # truncated list fragment (starts with a bullet/number marker,
        # e.g. "1. First item" cut off from a longer list) -- NOT just
        # any short answer. A short, correct factual answer (a code, a
        # number, a name) must not be overwritten by an unrelated list
        # pulled from elsewhere in the context. This was found via the
        # evaluation harness: "What is the O*NET-SOC code?" (a correct,
        # short answer) was being replaced by an unrelated bulleted list
        # under the old, overly broad "any short answer" rule.
        looks_like_truncated_list_item = bool(
            _NUMBERED_PATTERN.match(answer.strip()) or _BULLETED_PATTERN.match(answer.strip())
        )
        if looks_like_truncated_list_item and len(answer.split()) <= _SHORT_ANSWER_WORD_THRESHOLD:
            extracted = extract_list(context)
            if extracted:
                answer = extracted

    label = confidence_label(float(scores[0]))

    return answer, context, label

def test_answer_question_does_not_override_correct_short_factual_answer(monkeypatch):
    # Regression test found via evals/run_eval.py: "What is the O*NET-SOC
    # code?" got a correct, short answer ("15-1252.00") from the LLM, but
    # the old safety net treated ANY short answer as a truncated list and
    # overwrote it with an unrelated bulleted list from elsewhere in the
    # context. The safety net must only fire when the LLM's own answer
    # itself looks like a truncated list fragment (starts with a bullet
    # or number marker), not for any short-but-correct factual answer.
    import src.services.generation as gen

    monkeypatch.setattr(gen, "generate_answer", lambda context, query: "15-1252.00")

    context_chunks = [
        "The O*NET-SOC code for Software Developers is 15-1252.00.",
        "- Innovation - A tendency to be inventive.\n- Adaptability - A tendency to be open to change.",
    ]
    scores = [0.9]

    answer, context, label = gen.answer_question(context_chunks, scores, "What is the O*NET-SOC code?")

    assert answer == "15-1252.00"


def test_answer_question_still_rescues_truncated_list_fragment(monkeypatch):
    # Complementary case: if the LLM's short answer DOES look like a
    # truncated list item (starts with a number/bullet marker), the
    # safety net should still rescue the full list.
    import src.services.generation as gen

    monkeypatch.setattr(gen, "generate_answer", lambda context, query: "1. First item only")

    context_chunks = [
        "1. First item only\n2. Second item\n3. Third item",
    ]
    scores = [0.9]

    answer, context, label = gen.answer_question(context_chunks, scores, "What is required?")

    assert "Second item" in answer
    assert "Third item" in answer