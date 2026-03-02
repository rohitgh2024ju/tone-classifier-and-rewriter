from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load trained model from results folder
model_path = "models/tone_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# test model
classifier = pipeline(
    "text-classification",
    model="./results",
    tokenizer="./results"
)

tests = [
    "Send me the report.",
    "Hey bro send me that file asap.",
    "I would appreciate if you could provide the document.",
    "Update me.",
    "Can you please send this when possible?",
    "Do it now.",
    "Kindly review the attached file.",
    "Great job on this!",
    "This must be completed immediately.",
    "Pursuant to policy this must be approved."
]

print("\n--- Test Predictions ---")
for text in tests:
    print(f"{text} -> {classifier(text)}")