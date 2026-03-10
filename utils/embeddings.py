from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("intfloat/e5-base-v2")

def generate_embeddings(texts):
    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings