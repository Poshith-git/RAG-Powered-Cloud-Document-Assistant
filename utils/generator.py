from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load once (important)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def generate_answer(context, question):

    # Limit context size (VERY IMPORTANT)
    context = context[:1500]

    prompt = f"""
You are a Software Engineering expert.

Using only the context provided below,
answer the question clearly and in detail.

If the question asks for differences, present the answer in bullet points.
If the question asks for explanation, give at least 5 clear points.

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
            max_new_tokens=400,        # 🔥 Increased
            min_length=80,             # 🔥 Forces longer output
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if len(answer.strip()) < 10:
        return "The answer is not clearly available in the provided document."


    return answer.strip()
