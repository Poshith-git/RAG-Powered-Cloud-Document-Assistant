import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource
def load_generator():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model


# -----------------------------
# Answer generation
# -----------------------------
def generate_answer(context, question):

    tokenizer, model = load_generator()

    # Limit context size for stability
    context = context[:2000]

    prompt = f"""
You are a document analysis assistant.

Answer the question strictly using the provided context.

If the answer is not clearly available in the context, say:
"The answer is not available in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if len(answer.strip()) < 10:
        return "The answer is not clearly available in the provided document."

    return answer.strip()