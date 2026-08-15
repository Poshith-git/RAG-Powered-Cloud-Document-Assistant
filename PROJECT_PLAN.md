# Project Plan — RAG-Powered Cloud Document Assistant

## Problem Statement

Support/operations analysts and job seekers reviewing long PDF documents (policy
documents, contracts, job postings) need to find specific facts quickly, without
reading the whole document and without trusting an ungrounded chatbot answer that
might invent information the document doesn't contain.

## Target User

A person who has one PDF document and wants specific answers from it — not a
general-purpose research assistant, not a multi-document knowledge base (v1 scope).

## Scope (v1)

- Single-document PDF upload and Q&A
- Grounded answers with citation-worthy retrieved context shown to the user
- A visible confidence signal
- Structural (list-type) questions answered by extraction, not generation
- A reusable evaluation harness and golden dataset for regression testing
- Structured request logging

## Explicit Non-Goals (v1)

- Multi-document or cross-document sessions
- Non-PDF sources (CSV/DOCX) — scoped as a documented P1 feature, not built
- A production-grade authentication/rate-limiting layer
- A hosted monitoring dashboard (logs are structured to support one later, not built)
- Legal/compliance advice generation — the system cites and summarizes document
  content, and does not interpret legal validity

## Architecture Summary

See `README.md`'s Architecture section and `RAG_Assistant_v2_Design_Document.docx`
Chapter 3 for the full production architecture diagram and repository structure.
In short: PDF → pdfplumber → list/heading-aware chunking → E5 embeddings → FAISS →
section-aware retrieval → hybrid (rule-based list extraction / FLAN-T5 generation)
→ confidence-labeled answer → structured log.

## Milestones (as actually executed)

| Week | Focus | Outcome |
|---|---|---|
| 0 | Hotfix live bugs found via code audit | Fixed `is_query` embeddings crash, hardcoded single-document list logic, deprecated Streamlit API, double-truncation bug |
| 1 | Repository restructure | `src/services/` layer, unit tests, CI (GitHub Actions) |
| 2 | Golden dataset + evaluation harness | 12-case dataset, `evals/run_eval.py`, baseline: 41.7% |
| 3 | Chunking/retrieval tuning against the eval set | 75.0% pass rate, 6 documented root-cause fixes |
| 4 | Structured logging + real-world hardening | JSON request logs, log viewer, 3 more bugs found via testing on a real document, PDF extraction library swap (pypdf → pdfplumber), generator capability A/B test (Ollama), flan-t5-large trade-off study (measured and rejected for deployment) |

## Known Risks

- **Small local generator's numeric-extraction limitation** — documented, measured,
  and understood (not silently ignored). Mitigation path identified (larger model)
  but rejected for the free deployment tier due to a measured 100x latency cost.
- **Confidence heuristic is lexical, not learned** — approximate by design; documented
  as a limitation rather than presented as a solved problem.
- **Single-document synthetic eval set** — the golden dataset is built against one
  public-domain document; real-world testing against a second, differently-structured
  document surfaced 3 additional bugs the synthetic set didn't cover. Documented as
  a reason to keep testing against varied real documents, not just the fixed set.

## Evaluation Plan

`evals/golden_dataset.jsonl` + `evals/run_eval.py`, run before and after any
retrieval/chunking/generation change, with results written to `evals/results/` for
traceability. See README's Evaluation Results section for the full history.

## Deployment Plan

Docker + Hugging Face Spaces (free CPU tier). `Dockerfile` builds from
`requirements.txt`; entry point is `src/ui/streamlit_app.py`. `GENERATOR_BACKEND`
must be `"flan-t5"` (not `"ollama"`) before deploying — Ollama is local-only.

## Future Work

- Multi-source ingestion with retrieval routing (CSV structured lookup vs. PDF
  semantic search) — fully scoped in the v2 design document, not yet built
- FastAPI layer over the existing service layer
- Reranking, hybrid BM25+dense search
- A narrower investigation into why flan-t5-large fixed one numeric-extraction
  failure but not another with an apparently similar structure (see README's
  Known Limitations)