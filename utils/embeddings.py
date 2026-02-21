from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------------------------
# Load E5 Model
# -------------------------------------------------
model = SentenceTransformer("intfloat/e5-base-v2")


def generate_embeddings(texts, is_query=False):
    """
    Generate embeddings using E5 model.
    E5 requires special prefixes:
    - 'query: ' for queries
    - 'passage: ' for documents
    """

    if is_query:
        texts = [f"query: {text}" for text in texts]
    else:
        texts = [f"passage: {text}" for text in texts]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings