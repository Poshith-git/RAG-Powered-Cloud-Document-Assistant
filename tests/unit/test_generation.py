from src.services.generation import extract_list, confidence_label, order_by_intent


def test_extract_list_finds_numbered_items():
    # extract_list now takes a list of chunks (section-aware), not a
    # single merged string -- see the Week 3 fix for why.
    chunks = ["Intro text.", "1. First point\n2. Second point\n3. Third point", "Outro text."]
    result = extract_list(chunks)
    assert result is not None
    assert "First point" in result
    assert "Second point" in result


def test_extract_list_returns_none_for_single_item():
    # Only one list-like line should NOT be treated as a real list --
    # avoids false positives on stray numbered sentences.
    chunks = ["Some text.", "1. Only one numbered line here.", "More text."]
    assert extract_list(chunks) is None


def test_extract_list_works_on_arbitrary_document_not_just_spiral_model():
    # Regression test for the original hardcoded "Spiral Model" bug --
    # this must work on a completely unrelated document.
    chunks = [
        "Benefits of remote work:\n"
        "1. Flexible schedule\n"
        "2. No commute\n"
        "3. Better work-life balance\n"
    ]
    result = extract_list(chunks)
    assert result is not None
    assert "Flexible schedule" in result


def test_extract_list_does_not_mix_lists_from_different_chunks():
    # Week 3 regression test: found via evals/run_eval.py that a Skills
    # question was returning a numbered Tasks item, because the old
    # extract_list() pooled list-like lines across ALL retrieved chunks
    # regardless of which section they came from. The fix makes
    # extraction section-aware: check each chunk independently, and
    # return one chunk's own complete, internally-consistent list.
    chunks = [
        "1. Confer with systems analysts, engineers, and programmers.\n2. Modify existing software.",
        "- Critical Thinking - Using logic and reasoning.\n- Active Learning - Understanding new information.\n- Writing - Communicating effectively.",
    ]
    result = extract_list(chunks)
    # Without a query, falls back to retrieval-ranked order (first chunk).
    assert "Confer with systems analysts" in result
    assert "Modify existing software" in result
    assert "Critical Thinking" not in result


def test_extract_list_uses_query_to_pick_correct_section_when_top_rank_is_wrong():
    # Week 3 regression test #2: after the first section-aware fix,
    # re-running the eval harness showed "List all the tasks..." started
    # returning the Skills list instead of Tasks, because the Skills
    # chunk happened to rank first by cosine similarity even though it
    # was the wrong section. Fix: score candidate chunks by lexical
    # overlap between the query and the chunk's FULL text (including a
    # section heading like "Tasks" that precedes the list items), not
    # just take whichever chunk ranked first.
    #
    # NOTE: includes a few realistic distractor chunks (not just the 2
    # chunks directly under test) -- with only 2 total chunks, the
    # generic-word threshold (a FRACTION of total chunks) degenerates,
    # since any word appearing in even 1 of 2 chunks already looks
    # "generic". Real documents have more chunks than that; this test
    # reflects a realistic count instead of an artificial edge case.
    chunks = [
        # Ranked first by retrieval, but wrong section for a "tasks" query.
        "Essential Skills\n- Writing - Communicating effectively.\n- Speaking - Talking to others.",
        # Ranked second, but the correct section -- its own heading
        # ("Tasks") shares vocabulary with the query.
        "Tasks\n1. Analyze user needs and software requirements.\n2. Modify existing software to correct errors.",
        "Knowledge Areas\n- Mathematics - Knowledge of arithmetic and statistics.",
        "Work Styles\n- Dependability - A tendency to be reliable and consistent.",
        "Education\nA bachelor's degree is typically required for this occupation.",
    ]
    result = extract_list(chunks, query="List all the tasks performed by software developers.")
    assert "Analyze user needs" in result
    assert "Modify existing software" in result
    assert "Writing" not in result
    
def test_confidence_label_thresholds():
    assert confidence_label(0.90) == "High"
    assert confidence_label(0.70) == "Medium"
    assert confidence_label(0.40) == "Low"


def test_confidence_label_downgrades_on_no_lexical_overlap():
    # Week 3 fix: cosine similarity alone was found (via the eval harness)
    # to stay high (0.79-0.80) even for out-of-scope questions, since
    # FAISS always returns a "closest" chunk regardless of true relevance.
    # A high score with zero shared vocabulary between the query and the
    # top chunk should be downgraded to Low rather than trusted as-is.
    query = "What certifications are required for this role?"
    unrelated_chunk = "Median wages in 2025 were $65.38 hourly and $135,980 annually."
    assert confidence_label(0.90, query=query, ordered_chunks=[unrelated_chunk]) == "Low"


def test_confidence_label_keeps_high_score_when_overlap_exists():
    query = "What is the median annual wage?"
    relevant_chunk = "Median wages in 2025 were $65.38 hourly and $135,980 annually."
    assert confidence_label(0.90, query=query, ordered_chunks=[relevant_chunk]) == "High"


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

def test_confidence_label_ignores_generic_document_wide_words():
    # Week 3 regression test: found via the eval harness that "Which
    # companies hire the most software developers?" was NOT being
    # refused, because "software" and "developers" appear in nearly
    # every chunk of a document that is entirely about software
    # developers -- so plain word overlap passed trivially, giving no
    # real refusal signal. Words appearing in most chunks must be
    # excluded from the overlap check as "generic to this document".
    query = "Which companies hire the most software developers?"
    chunks = [
        "Software Developers (O*NET-SOC Code 15-1252.00)",
        "Research, design, and develop computer and network software for software developers.",
        "Tasks performed by software developers include analyzing user needs.",
    ]
    assert confidence_label(0.90, query=query, ordered_chunks=chunks) == "Low"    

def test_word_overlap_does_not_false_positive_on_shared_prefix():
    # Regression test for a real bug found via the eval harness: fixed-
    # length 4-char prefix truncation caused "companies" and "computers"
    # to both truncate to "comp", spuriously matching them and defeating
    # the out-of-scope refusal check for "Which companies hire..." on a
    # document that mentions "computers" everywhere. Prefix-containment
    # matching (is one word a genuine prefix of the other?) must reject
    # this pair while still accepting real variants like wage/wages.
    from src.services.generation import _words_match
    assert not _words_match("companies", "computers")
    assert not _words_match("companies", "computer")
    assert _words_match("wage", "wages")
    assert _words_match("annual", "annually")


def test_confidence_label_clusters_word_families_as_generic():
    # Regression test found via a diagnostic script after "Which
    # companies hire the most software developers?" still wasn't being
    # refused even after the prefix-collision fix above. Root cause:
    # exact-token document-frequency counting saw "developers" (query)
    # as a separate, rare token from "develop" (appearing in the
    # Overview section) -- missing that they're the same word family --
    # so "developers" wasn't recognized as generic to a document
    # entirely about software developers, and matched the Overview
    # chunk's "develop" via prefix-containment, defeating refusal.
    # Genericity must be computed via the same word-family clustering
    # used for the overlap check itself, not exact-token frequency.
    query = "Which companies hire the most software developers?"
    chunks = [
        "Software Developers (O*NET-SOC Code 15-1252.00)",
        "Research, design, and develop computer and network software for various industries.",
        "Tasks performed include analyzing user needs and requirements.",
    ]
    assert confidence_label(0.90, query=query, ordered_chunks=chunks) == "Low"    