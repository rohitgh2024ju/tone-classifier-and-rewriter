Tone Classifier and Rewriter

Overview--

This project is an end-to-end NLP application that performs:

Tone classification of input text using a fine-tuned transformer model

Tone rewriting using a large language model (LLM)

The system combines a locally trained DistilBERT-based classifier with an external generative model to produce high-quality tone-adjusted text while preserving original meaning.

Features--

Multi-class tone classification

Top prediction confidence scores

Tone-aware text rewriting (3 variations)

Interactive web interface (Streamlit)

REST API backend (FastAPI)

Modular and scalable architecture--

Supported Tones

Casual

Professional

Polite

Friendly

Assertive

Formal

Model Details--
Base Model-

The classifier is built on:

DistilBERT (transformer-based encoder)

Fine-tuned for sequence classification

Architecture-

Input: raw text

Tokenization: Hugging Face tokenizer

Encoder: DistilBERT

Classification head: linear layer over pooled output

Output: probability distribution over tone classes

Training Setup-

Dataset: custom tone-labeled dataset

Loss function: cross-entropy loss

Optimization: AdamW

Evaluation metrics:

Accuracy: ~88%

F1-score: ~88%

Output Format-

The model returns:

Predicted tone

Confidence scores for each class

Top-2 predictions (used in UI)

Rewriting Module--

Tone rewriting is handled using an external LLM via Groq API.

Model Used-

LLaMA 3.3 (70B, versatile)

Prompt Design-

The prompt enforces:

Preservation of original meaning

Exactly 3 variations

One sentence per variation

No additional explanation

Controlled tone transformation

System Architecture--
User (Browser)
│
▼
Streamlit UI (Frontend)
│
▼
FastAPI Backend (/predict)
│
├── Tone Classification (DistilBERT)
│
└── Tone Rewriting (Groq LLM API)
│
▼
Response (JSON)
│
▼
UI Rendering

API Specification--
Endpoint-

POST /predict

Request Body-
{
"text": "input sentence",
"target_tone": "Professional"
}
Response-
{
"text": "...",
"detected_tone": "...",
"target_tone": "...",
"predictions": [
{"label": "...", "confidence": ...}
],
"suggestions": [
"...",
"...",
"..."
]
}
Frontend--

Built with Streamlit

Custom CSS for modern UI

Interactive elements:

tone display

confidence bars

suggestion cards

copy-to-clipboard functionality

Deployment Architecture--

Current setup:

Frontend: Streamlit Cloud

Backend: Local FastAPI server exposed via ngrok

User → Streamlit Cloud → ngrok → Local FastAPI → Model

Limitations-

Backend depends on local machine

ngrok URL changes on restart

not suitable for production

Future Improvements--

Deploy backend on cloud (Render / Railway / DigitalOcean)

Replace ngrok with permanent API endpoint

Add authentication and rate limiting

Improve dataset size and class balance

Experiment with larger or domain-specific transformer models

Build dedicated frontend (React) for full UI control

Project Structure--
tone_analyzer/
│
├── app.py # Streamlit UI
├── model_api.py # FastAPI backend
├── model_train.py # training script
├── models/
│ └── tone_model/ # saved transformer model
├── requirements.txt
├── Procfile
└── README.md

Key Learnings--

Fine-tuning transformer models for classification tasks

Designing prompt-controlled text generation

Building REST APIs for ML models

Handling deployment constraints for large models

Managing frontend-backend integration

Conclusion--

This project demonstrates a complete ML application pipeline:

Model training

API development

LLM integration

UI design

Deployment
