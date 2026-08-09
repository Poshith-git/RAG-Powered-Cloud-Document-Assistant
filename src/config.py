"""
Central configuration for the RAG pipeline.
Pulling these out of inline code makes them easy to tune during
evaluation (Week 2) instead of hunting through app.py for magic numbers.
"""

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 8
# Was 2000 but never actually wired into utils/generator.py, which
# independently hardcoded context[:2000] -- found via Week 3 tuning
# against the eval harness: the Wages section (short, numbers-dense,
# doesn't rank in the top few chunks by similarity) was being silently
# cut off before the LLM ever saw it. Raised and now actually used.
MAX_CONTEXT_CHARS = 2000
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 250

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.65