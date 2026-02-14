import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def search_index(index, query_embedding, chunks, top_k=3):
    """
    Retrieve top-k most similar chunks.
    """
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])

    return results
