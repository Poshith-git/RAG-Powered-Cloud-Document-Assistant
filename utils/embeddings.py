from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("intfloat/e5-base-v2")

def generate_embeddings(texts, is_query=False):
    """
    E5 models require a 'query: ' or 'passage: ' prefix on the raw text
    to produce correctly-aligned embeddings -- without this, retrieval
    quality silently degrades even though no error is raised.
    """
    model = load_embedding_model()
    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [prefix + t for t in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True)
    return embeddings