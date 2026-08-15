Every box above exists because a real bug was found and fixed there — see
[Evaluation Results](#evaluation-results) and the design document for the full
before/after history.

## Features

- **PDF upload and Q&A** via a Streamlit UI
- **Grounded answers**: retrieval happens before generation; the model is instructed
  to say the answer isn't available rather than guess
- **Hybrid answering**: structural questions ("what are the qualifications") are
  answered by extracting the actual list from the document, not by asking a small
  LLM to reproduce it (which was found to truncate or hallucinate)
- **Confidence signal**: High / Medium / Low, based on both retrieval similarity and
  whether the query shares real vocabulary with the retrieved content
- **Structured request logging**: every request is logged as JSON
  (`logs/requests.jsonl`), viewable via `scripts/view_logs.py`
- **Swappable generator backend**: FLAN-T5-base by default (free, deployable on
  Hugging Face Spaces' CPU tier); an Ollama backend for local testing with a larger
  model (see [Generator Trade-off Study](#generator-trade-off-study))

## Demo

Try the live application on Hugging Face Spaces:
https://huggingface.co/spaces/Manchivishyam/rag-document-assistant

## Installation

```bash
git clone <this-repo>
cd RAG-Powered-Cloud-Document-Assistant
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements-dev.txt
```

## Running

```bash
streamlit run src/ui/streamlit_app.py
```

Upload a PDF, then ask a question.

## Evaluation Results

A golden dataset (`evals/golden_dataset.jsonl`, 12 cases across easy/list/hard/
paraphrase/out-of-scope/citation categories) is run against the system with
`evals/run_eval.py`, against a public-domain sample PDF (`data/sample_pdfs/`, built
from U.S. Department of Labor O*NET data — no private documents are committed to
this repo).

| Stage | Pass rate | What changed |
|---|---|---|
| Week 2 baseline | 41.7% (5/12) | First harness run; caught a live bug (a safety-net override was replacing correct short answers with unrelated lists) |
| Week 3 tuning | 75.0% (9/12) | Six fixes: FAISS `-1` padding bug, section-aware list extraction, heading-attachment in chunking, prefix-collision false positive, word-family generic-word clustering, threshold tuning against real diagnostic data |
| flan-t5-large (measured, not deployed) | 83.3% (10/12) | Confirmed remaining failures were a generator capability limitation — reverted due to ~100x latency increase on CPU (see below) |

Run it yourself:
```bash
python evals/run_eval.py --pdf data/sample_pdfs/software_developers_onet_summary.pdf
```

### Generator Trade-off Study

Two wage/growth-figure questions in the golden dataset failed consistently even
though retrieval correctly surfaced the right content with high confidence. A
controlled A/B test (same document, same fact, only the generator changed) confirmed
this was a genuine FLAN-T5-base (250M param) capability limitation: swapping to a
local `qwen3:14b` model via an optional Ollama backend fixed the failure completely
and consistently across phrasings.

A same-architecture upgrade (`flan-t5-large`, 780M params) was then tried as a
free, still-deployable option: it raised the eval pass rate to 83.3%, but average
CPU latency rose from ~850ms to ~73,300ms — roughly 100x slower, not the ~3x its
parameter count would suggest. That's unusable for a live demo, so the deployed
default remains `flan-t5-base`. This is documented in
`RAG_Assistant_Week4_Addendum.docx` and in `src/config.py`'s comments, as a
measured, rejected-for-production decision rather than an untested assumption.

## Known Limitations (v1)

Documented honestly rather than hidden:

- **Small-model numeric extraction**: FLAN-T5-base can miss a specific number in a
  numbers-dense passage, particularly when the passage contains multiple competing
  figures close together (see Generator Trade-off Study above).
- **Confidence calibration is a heuristic, not a classifier**: it uses lexical
  word-overlap with document-generic-word filtering, which is an approximation —
  it can still be fooled by unusual phrasing or very short documents.
- **List extraction inside a single run-on paragraph is imperfect**: query-anchored
  windowing mitigates but doesn't fully solve documents where multiple sections have
  zero separating structure (rare after the `pdfplumber` extraction fix, but possible).
- **Single-document sessions only**: no cross-document Q&A yet (see Roadmap).

## Roadmap

- [ ] Multi-source ingestion (CSV/DOCX) with retrieval routing between semantic and
      structured lookup (see `RAG_Assistant_v2_Design_Document.docx`, Chapter 4.2.1)
- [ ] FastAPI layer behind the existing service layer (already framework-agnostic)
- [ ] Reranking of top-k candidates before generation
- [ ] Hosted monitoring dashboard (logs are already structured for this)
- [ ] Request rate-limiting and prompt-injection sanitization for untrusted PDF text

## Tests

```bash
pytest tests/unit tests/integration -v
```
38 tests, covering chunking, retrieval, generation logic, list extraction,
confidence calibration, structured logging, and upload validation — each with a
comment explaining which real bug it's a regression test for.

## Tech Stack

Python, Streamlit, FAISS, Hugging Face Transformers (`intfloat/e5-base-v2`,
`google/flan-t5-base`), pdfplumber, pytest, GitHub Actions (CI), Docker,
Hugging Face Spaces.

## Project History

This project went through a full engineering review and transformation, documented
in `RAG_Assistant_v2_Design_Document.docx` (architecture, evaluation framework,
resume/interview prep) and `RAG_Assistant_Week4_Addendum.docx` (real-world bug
findings, generator trade-off study, security hardening, final scorecard). Both are
worth reading before an interview about this project — they contain the actual
before/after evidence for every engineering decision above, not just the end state.

## License

MIT License