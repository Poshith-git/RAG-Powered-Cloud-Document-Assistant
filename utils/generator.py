from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# -------------------------------------------------
# Load Model (FLAN-T5-base)
# -------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


# -------------------------------------------------
# Detect Question Type
# -------------------------------------------------
def detect_question_type(question):
    q = question.lower()

    # Force list mode for advantages/disadvantages
    if "advantages" in q or "disadvantages" in q:
        return "list"

    if any(word in q for word in ["list", "all"]):
        return "list"

    if any(word in q for word in ["explain", "describe", "why", "how", "compare"]):
        return "explanation"

    return "short"


# -------------------------------------------------
# Build Prompt
# -------------------------------------------------
def build_prompt(context, question, question_type):

    if question_type == "list":
        instruction = """
Extract every numbered point from the context.
Preserve numbering exactly.
Do NOT summarize.
Do NOT merge points.
Return each point as a clearly separated numbered item.
"""
    elif question_type == "short":
        instruction = """
Provide a concise definition in 2–3 sentences only.
"""
    else:
        instruction = """
Provide a structured explanation without repeating sentences.
"""

    return f"""
You are a precise document analysis assistant.

Use ONLY the information provided in the context below.
If the answer is not explicitly present in the context, respond with:
"The answer is not available in the document."

{instruction}

Context:
{context}

Question:
{question}

Answer:
"""


# -------------------------------------------------
# Generate Answer
# -------------------------------------------------
def generate_answer(context, question):

    context = context[:1500]

    question_type = detect_question_type(question)
    prompt = build_prompt(context, question, question_type)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Adaptive token allocation
    if question_type == "list":
        max_tokens = 450   # Give full room for all list items
    elif question_type == "explanation":
        max_tokens = 300
    else:
        max_tokens = 120

    # Context scaling (ONLY for non-list types)
    if question_type != "list":
        if len(context) < 500:
            max_tokens = min(max_tokens, 150)
        elif len(context) < 1000:
            max_tokens = min(max_tokens, 250)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )


    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not answer:
        return "The answer is not clearly available in the provided document."

    # ------------------------------
    # Clean Formatting Layer
    # ------------------------------

    # Fix spacing issues
    answer = answer.replace("  ", " ")

    # Add newline before numbered items
    import re
    answer = re.sub(r"(\d+\.)", r"\n\1", answer)

    # Remove accidental multiple newlines
    answer = re.sub(r"\n{2,}", "\n", answer)

    return answer.strip()