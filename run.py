import os
from transformers import (
    pipeline,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from groq import Groq
from dotenv import load_dotenv

# --------- Load ENV ---------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# --------- Load Model ---------
model = DistilBertForSequenceClassification.from_pretrained("./models/tone_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./models/tone_model")

classifier = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, top_k=None
)


# --------- Rewrite Function ---------
def rewrite(text, source_tone, target_tone):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You rewrite sentences into different tones while preserving meaning.",
            },
            {
                "role": "user",
                "content": f"""
Rewrite the following sentence from {source_tone} tone to {target_tone} tone.

Rules:
- Generate EXACTLY 3 variations
- Keep each variation concise (1 sentence)
- Do not add extra meaning
- Do not explain anything
- Do not change the intent or type of question
- Preserve whether the sentence is asking identity, action, or confirmation
- Output as a numbered list
- Do not add new emotions or opinions
- Do not introduce enthusiasm unless already present
- Keep the sentence factual if the original is factual

Sentence:
{text}
""",
            },
        ],
        temperature=0.6,
    )

    output = response.choices[0].message.content.strip()

    # parse output
    lines = output.split("\n")
    suggestions = []

    for line in lines:
        line = line.strip()
        if line:
            cleaned = line.lstrip("1234567890. ").strip()
            suggestions.append(cleaned)

    return suggestions[:3]


# --------- CLI LOOP ---------
def main():
    print("\nTone Analyzer CLI (type 'exit' to quit)\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() == "exit":
            print("Exiting...")
            break

        target_tone = input(
            "Target tone (Casual / Professional / Polite / Friendly / Assertive / Formal): "
        ).strip()

        if not target_tone:
            target_tone = "Professional"

        # --------- Classification ---------
        result = classifier(text)[0]
        sorted_preds = sorted(result, key=lambda x: x["score"], reverse=True)
        top2 = sorted_preds[:2]

        detected_tone = top2[0]["label"]

        print("\nDetected Tone:", detected_tone)
        print("Top Predictions:")
        for p in top2:
            print(f"  - {p['label']}: {round(p['score']*100, 2)}%")

        # --------- Rewrite ---------
        suggestions = rewrite(text, detected_tone, target_tone)

        print("\nSuggestions:")
        for i, s in enumerate(suggestions, 1):
            print(f"{i}. {s}")

        print("\n" + "-" * 50 + "\n")


# --------- Run ---------
if __name__ == "__main__":
    main()
