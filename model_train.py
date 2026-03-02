import os
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------- Load Dataset ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "tone_dataset.csv")

df = pd.read_csv(file_path)

# --------- Label Mapping ---------
label2id = {
    "Casual": 0,
    "Professional": 1,
    "Polite": 2,
    "Friendly": 3,
    "Assertive": 4,
    "Formal": 5
}

id2label = {v: k for k, v in label2id.items()}

# Filter + map labels
df = df[df["label"].isin(label2id.keys())]
df["label"] = df["label"].map(label2id)

print("Total samples:", len(df))
print(df["label"].value_counts())

# --------- Train / Validation Split ---------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# --------- Tokenizer ---------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=160
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# --------- Model ---------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6
)

model.config.id2label = id2label
model.config.label2id = label2id

# --------- Metrics ---------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# --------- Training Arguments ---------
training_args = TrainingArguments(
    output_dir="./results_bert",
    num_train_epochs=3,
    per_device_train_batch_size=8,   # reduce to 4 if memory issue
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=20,
    report_to="none",
)

# --------- Trainer ---------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --------- Train ---------
trainer.train()

# --------- Evaluate ---------
results = trainer.evaluate()
print("\nFinal Evaluation:", results)

# --------- Save Model ---------
trainer.save_model("./results_bert")
tokenizer.save_pretrained("./results_bert")

# --------- Test Pipeline ---------
classifier = pipeline(
    "text-classification",
    model="./results_bert",
    tokenizer="./results_bert"
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