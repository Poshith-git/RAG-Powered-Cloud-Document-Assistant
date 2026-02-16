import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------
# Model Configuration
# ----------------------------------------
MODEL_NAME = "google/flan-t5-small"

HF_TOKEN = os.getenv("HF_TOKEN")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

model.eval()


# ----------------------------------------
# Answer Generation Function
# ----------------------------------------
def generate_answer(context, question):
    """
    Generates answer using FLAN-T5-small model.
    """

    if not context.strip():
        return "No relevant context found in the document."

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the provided context.

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
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean output
    answer = decoded.strip()

    if answer == "":
        return "The model could not generate a proper answer."

    return answer
