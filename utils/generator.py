import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from src.config import MAX_CONTEXT_CHARS, MAX_NEW_TOKENS, MAX_INPUT_TOKENS, GENERATOR_MODEL_NAME


@st.cache_resource
def load_generator():
    # Wired to config.GENERATOR_MODEL_NAME instead of a hardcoded
    # string -- previously this was hardcoded to "google/flan-t5-base"
    # independent of src/config.py's GENERATOR_MODEL_NAME constant,
    # which existed but was never actually used. Swapping models
    # (e.g. to flan-t5-large) is now a one-line config change instead
    # of touching this file.
    tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL_NAME)
    return tokenizer, model


def generate_answer(context, question):

    tokenizer, model = load_generator()

    context = context[:MAX_CONTEXT_CHARS]

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
        max_length=MAX_INPUT_TOKENS
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if len(answer.strip()) < 10:
        return "The answer is not clearly available in the provided document."

    return answer.strip()