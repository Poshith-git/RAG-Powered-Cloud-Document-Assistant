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
MAX_CONTEXT_CHARS = 2000
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 250

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.65