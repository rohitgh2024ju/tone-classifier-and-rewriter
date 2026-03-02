from transformers import (
    pipeline,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
from dotenv import load_dotenv

# --------- Load API Key (IMPORTANT: do not hardcode) ---------
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

# --------- Load Model ---------
BASE_DIR = os.path.dirname(__file__)

model = DistilBertForSequenceClassification.from_pretrained("./results")
tokenizer = DistilBertTokenizerFast.from_pretrained("./results")

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,  # return all class scores
)


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
- Generate EXACTLY 3 different variations
- Keep each variation concise (1 sentence)
- Do not add extra meaning
- Do not explain anything
- Keep original intent
- Output as a numbered list
- Preserve specific wording when possible

Sentence:
{text}
""",
                },
            ],
            temperature=0.6,  # slightly higher for diversity
        )

        output = response.choices[0].message.content.strip()

        # parse numbered output into list
        lines = output.split("\n")
        suggestions = []

        for line in lines:
            line = line.strip()
            if line:
                cleaned = line.lstrip("1234567890. ").strip()
                suggestions.append(cleaned)

        return suggestions[:3]

    except Exception as e:
        return [f"Rewrite failed: {str(e)}"]


# --------- Input Schema ---------
class Profile(BaseModel):
    text: str
    target_tone: str = "Professional"


# --------- Prediction Route ---------
@app.post("/predict")
def predict(data: Profile):
    try:
        # run classifier
        result = classifier(data.text)[0]

        # sort predictions
        sorted_preds = sorted(result, key=lambda x: x["score"], reverse=True)

        # top 2 predictions
        top2 = sorted_preds[:2]

        detected_tone = top2[0]["label"]

        # rewrite
        rewritten = rewrite(
            text=data.text,
            source_tone=detected_tone,
            target_tone=data.target_tone,
        )

        return {
            "text": data.text,
            "detected_tone": detected_tone,
            "target_tone": data.target_tone,
            "predictions": [
                {
                    "label": p["label"],
                    "confidence": round(p["score"] * 100, 2),
                }
                for p in top2
            ],
            "suggestions": rewritten,
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input")

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
