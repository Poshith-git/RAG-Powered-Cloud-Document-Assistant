from src.services.generation import extract_list, confidence_label, order_by_intent


def test_extract_list_finds_numbered_items():
    context = "Intro text.\n1. First point\n2. Second point\n3. Third point\nOutro text."
    result = extract_list(context)
    assert result is not None
    assert "First point" in result
    assert "Second point" in result


def test_extract_list_returns_none_for_single_item():
    # Only one list-like line should NOT be treated as a real list --
    # avoids false positives on stray numbered sentences.
    context = "Some text.\n1. Only one numbered line here.\nMore text."
    assert extract_list(context) is None


def test_extract_list_works_on_arbitrary_document_not_just_spiral_model():
    # Regression test for the original hardcoded "Spiral Model" bug --
    # this must work on a completely unrelated document.
    context = (
        "Benefits of remote work:\n"
        "1. Flexible schedule\n"
        "2. No commute\n"
        "3. Better work-life balance\n"
    )
    result = extract_list(context)
    assert result is not None
    assert "Flexible schedule" in result


def test_confidence_label_thresholds():
    assert confidence_label(0.90) == "High"
    assert confidence_label(0.70) == "Medium"
    assert confidence_label(0.40) == "Low"


def test_order_by_intent_boosts_definitions():
    chunks = ["Random unrelated text.", "A cat is a small domesticated mammal."]
    ordered = order_by_intent(chunks, "what is a cat")
    assert ordered[0] == "A cat is a small domesticated mammal."

def test_looks_like_list_question_catches_qualifications():
    # Regression test: "what are the qualifications required" was
    # previously missed by the fixed 3-word trigger list, causing the
    # LLM to truncate a full list down to one bullet.
    from src.services.generation import _looks_like_list_question
    assert _looks_like_list_question("what are the qualifications required for this role?")
    assert _looks_like_list_question("what are the responsibilities of this job?")
    assert _looks_like_list_question("what benefits are offered?")    