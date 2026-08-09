import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create FAISS index using cosine similarity.
    """
    embeddings = np.array(embeddings).astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

    index.add(embeddings)

    return index


def search_index(index, query_embedding, chunks, top_k=5):
    """
    Retrieve top-k most similar chunks using cosine similarity.
    Returns chunks sorted by similarity (highest first).
    """

    # FAISS pads results with index -1 when top_k exceeds the number of
    # vectors in the index. Since Python allows negative indexing,
    # chunks[-1] silently resolves to the LAST chunk instead of raising
    # an error -- for small documents (fewer chunks than TOP_K), this
    # was duplicating the last chunk into the results multiple times,
    # which skewed every document-frequency-based calculation downstream
    # (found via the eval harness debug script). Clamp top_k so FAISS
    # never has to pad.
    effective_top_k = min(top_k, index.ntotal)

    query_embedding = np.array([query_embedding]).astype("float32")

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, effective_top_k)

    results = []

    for i, score in zip(indices[0], distances[0]):
        if i == -1:
            continue
        results.append((chunks[i], float(score)))

    # Ensure sorted by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    final_chunks = [r[0] for r in results]
    final_scores = [r[1] for r in results]

    return final_chunks, final_scores