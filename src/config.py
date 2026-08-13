"""
Central configuration for the RAG pipeline.
Pulling these out of inline code makes them easy to tune during
evaluation (Week 2) instead of hunting through app.py for magic numbers.
"""

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

# Swappable generator backend: "flan-t5" (default, free-tier deployable
# on Hugging Face Spaces) or "ollama" (local dev/testing only -- Ollama
# can't run on the free HF Space, no GPU/resources there, and the model
# isn't bundled). Kept swappable via one config value rather than
# hard-replacing FLAN-T5, since a local, more capable model is genuinely
# useful for testing whether generator quality (not retrieval/chunking)
# is the root cause of a failure -- confirmed via a real A/B test: the
# exact same wage-extraction question that failed with FLAN-T5-base
# under one phrasing succeeded consistently under both phrasings once
# swapped to a local qwen3:14b model via Ollama, with retrieval/
# confidence scores unchanged -- proving the failure was a generator
# capability limitation, not a retrieval or chunking bug.
GENERATOR_BACKEND = "flan-t5"  # or "ollama"

# Upgraded to flan-t5-large (780M params) during testing based on A/B
# evidence that the wage/growth-figure extraction failures were a
# generator capability limitation -- measured result: eval pass rate
# rose from 75.0% to 83.3%. But average latency on CPU went from ~850ms
# to ~73,300ms (roughly 100x slower, not the ~3x its parameter count
# would suggest) -- unusable for a live demo on the free Hugging Face
# Spaces CPU tier, where a 70+ second wait per question would look
# hung or hit request timeouts. Reverted to flan-t5-base for
# deployment based on that measurement, the same way MAX_CONTEXT_CHARS
# was reverted earlier when a change helped one metric while hurting
# another. flan-t5-large remains a documented, measured option for
# contexts where latency isn't constrained (batch processing, a paid
# GPU tier, or local testing via GENERATOR_BACKEND="ollama" instead,
# which is faster in practice despite being a larger model, since it
# runs against a properly optimized local inference server rather than
# raw CPU transformers).
GENERATOR_MODEL_NAME = "google/flan-t5-base"

OLLAMA_MODEL_NAME = "qwen3:14b"
OLLAMA_HOST = "http://localhost:11434"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 8
MAX_CONTEXT_CHARS = 2000
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 250

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.65