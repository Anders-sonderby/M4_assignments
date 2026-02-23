import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── Load data ─────────────────────────────────────────────────
train_silver = pd.read_csv("train_silver.csv")
eval_silver  = pd.read_csv("eval_silver.csv")
gold_100     = pd.read_csv("hitl_final.csv")
print(f"Train silver: {len(train_silver)}, Eval: {len(eval_silver)}, Gold: {len(gold_100)}")

# ── Prepare labels ────────────────────────────────────────────
train_silver["label"] = train_silver["is_green_silver"].astype(int)
eval_silver["label"]  = eval_silver["is_green_silver"].astype(int)
gold_100["label"]     = gold_100["is_green_gold"].astype(int)

# ── Combine train_silver + gold_100 ───────────────────────────
train_combined = pd.concat([
    train_silver[["text", "label"]],
    gold_100[["text", "label"]]
], ignore_index=True)
print(f"Combined training set: {len(train_combined)}")

# ── HuggingFace datasets ──────────────────────────────────────
train_ds = Dataset.from_pandas(train_combined[["text", "label"]], preserve_index=False)
eval_ds  = Dataset.from_pandas(eval_silver[["text", "label"]],   preserve_index=False)

# ── Tokenizer ─────────────────────────────────────────────────
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds  = eval_ds.map(tokenize,  batched=True)

# ── Model ──────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ── Metrics ───────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ── Training arguments ────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./patentsbert_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,
)

# ── Trainer ───────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ── Save model + tokenizer ────────────────────────────────────
trainer.save_model("./patentsbert_finetuned_final")
tokenizer.save_pretrained("./patentsbert_finetuned_final")
print("Model saved.")

# ── Save eval predictions ─────────────────────────────────────
predictions_output = trainer.predict(eval_ds)
preds = np.argmax(predictions_output.predictions, axis=-1)

eval_results_df = eval_silver[["text", "label"]].copy().reset_index(drop=True)
eval_results_df["predicted_label"] = preds
eval_results_df["correct"] = eval_results_df["label"] == eval_results_df["predicted_label"]
eval_results_df.to_csv("./eval_predictions.csv", index=False)
print("Eval predictions saved.")

# ── Save metrics ──────────────────────────────────────────────
metrics_df = pd.DataFrame([predictions_output.metrics])
metrics_df.to_csv("./eval_metrics.csv", index=False)
print("Metrics saved.")
print(metrics_df)