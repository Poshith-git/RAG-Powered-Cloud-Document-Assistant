from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_answer(context, question):
    """
    Generate structured answer using FLAN-T5-small.
    """

    prompt = f"""
You are a professional technical assistant.

Answer the question clearly and in complete sentences.
Use only the information provided in the context.
Do not repeat phrases.
Do not copy section numbers.
Do not list bullet points.
Write a short paragraph explanation.

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
            max_new_tokens=200,
            temperature=0.2,      # reduce randomness
            repetition_penalty=1.2,  # reduce repetition loops
            num_beams=4           # more structured output
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip()
