"""
Retrieval service: embeds chunks/queries and runs FAISS search.
"""

from utils.embeddings import generate_embeddings
from utils.retriever import create_faiss_index, search_index
from src.config import TOP_K


def build_index(chunks):
    """Embed a list of chunks (as passages) and build a FAISS index."""
    embeddings = generate_embeddings(chunks, is_query=False)
    index = create_faiss_index(embeddings)
    return index, embeddings


def retrieve(index, chunks, query, top_k=TOP_K):
    """Embed a query and retrieve the top-k most relevant chunks."""
    query_embedding = generate_embeddings([query], is_query=True)[0]
    results, scores = search_index(index, query_embedding, chunks, top_k=top_k)
    return results, scores