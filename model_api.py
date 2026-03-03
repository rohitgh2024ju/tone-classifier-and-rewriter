from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
from dotenv import load_dotenv

# --------- Load Environment ---------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment")

client = Groq(api_key=GROQ_API_KEY)

# --------- App ---------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Load Model (robust path) ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "rohit2004ju/tone-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# --------- Prediction Function (no pipeline for better control) ---------
import torch


def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze()

    labels = ["Casual", "Professional", "Polite", "Friendly", "Assertive", "Formal"]

    results = []
    for i, prob in enumerate(probs):
        results.append({"label": labels[i], "confidence": round(prob.item() * 100, 2)})

    # sort descending
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    return results


# --------- Rewrite Function ---------
def rewrite(text, source_tone, target_tone):
    try:
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
- Keep each to one sentence
- No explanations
- Output numbered list
- Do not change the intent or type of question
- Preserve whether the sentence is asking identity, action, or confirmation
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

        suggestions = []
        for line in output.split("\n"):
            line = line.strip()
            if line:
                cleaned = line.lstrip("1234567890. ").strip()
                suggestions.append(cleaned)

        return suggestions[:3]

    except Exception:
        return ["Rewrite unavailable at the moment."]


# --------- Input Schema ---------
class Profile(BaseModel):
    text: str
    target_tone: str = "Professional"


# --------- Prediction Route ---------
@app.post("/predict")
def predict(data: Profile):
    try:
        predictions = classify(data.text)

        detected_tone = predictions[0]["label"]

        rewritten = rewrite(
            text=data.text,
            source_tone=detected_tone,
            target_tone=data.target_tone,
        )

        return {
            "text": data.text,
            "detected_tone": detected_tone,
            "target_tone": data.target_tone,
            "predictions": predictions[:2],
            "suggestions": rewritten,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- Health Route ---------
@app.get("/")
def home():
    return {
        "message": "Tone Classifier API (DistilBERT, 6 classes)",
        "labels": [
            "Casual",
            "Professional",
            "Polite",
            "Friendly",
            "Assertive",
            "Formal",
        ],
    }
