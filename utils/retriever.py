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
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)

    index.add(embeddings)

    return index


def search_index(index, query_embedding, chunks, top_k=3):
    """
    Retrieve top-k most similar chunks.
    """
    query_embedding = np.array([query_embedding]).astype("float32")

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results, distances[0]

