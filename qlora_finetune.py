"""
Final Assignment - Part C, Step 1
QLoRA Fine-tuning of Mistral-7B on patent claims (train_silver.csv)
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# ── 1. HuggingFace login ──────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Run: export HF_TOKEN=your_token")
login(token=HF_TOKEN)
print("✅ Logged in to HuggingFace")

# ── 2. Load train_silver ──────────────────────────────────────────────────────
DATA_PATH = "/ceph/home/student.aau.dk/as58zr/M4_assignments/train_silver.csv"
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

df = df[["text", "is_green_silver"]].dropna()
print(f"✅ Loaded {len(df)} rows")
print(f"   Label distribution:\n{df['is_green_silver'].value_counts().to_string()}")

# ── 3. Format as instruction-tuning prompts ───────────────────────────────────
def format_prompt(row):
    label_str = "YES" if int(row["is_green_silver"]) == 1 else "NO"
    return {
        "text": (
            "### Task: Classify the following patent claim as green technology (Y02) or not.\n\n"
            f"### Claim:\n{str(row['text'])[:512]}\n\n"
            f"### Answer: {label_str}"
        )
    }

print("Formatting prompts...")
formatted = [format_prompt(row) for _, row in df.iterrows()]
hf_dataset = Dataset.from_list(formatted).train_test_split(test_size=0.05, seed=42)
print(f"✅ Train size: {len(hf_dataset['train'])} | Eval size: {len(hf_dataset['test'])}")

# ── 4. Load Mistral-7B in 4-bit (QLoRA) ──────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-v0.1"
print(f"\nLoading {MODEL_ID} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
print("✅ Model loaded and prepared for k-bit training")

# ── 5. LoRA configuration ─────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
print(f"✅ LoRA config set — r={lora_config.r}, alpha={lora_config.lora_alpha}")

# ── 6. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./qlora-mistral-patent-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    warmup_steps=50,
    lr_scheduler_type="cosine",
)

# ── 7. Trainer ────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["test"],
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("\n🚀 Starting QLoRA fine-tuning...")
trainer.train()
print("✅ Training complete!")

# ── 8. Save eval metrics locally ──────────────────────────────────────────────
metrics = trainer.evaluate()
print(f"\nFinal eval metrics: {metrics}")

metrics_df = pd.DataFrame([metrics])
metrics_path = "/ceph/home/student.aau.dk/as58zr/M4_assignments/qlora_eval_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"✅ Eval metrics saved to {metrics_path}")

# ── 9. Push to HuggingFace Hub ────────────────────────────────────────────────
HF_REPO = "mistral-7b-patent-qlora"
print(f"\nPushing model to HuggingFace Hub as {HF_REPO}...")
trainer.model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"✅ Model pushed to: https://huggingface.co/{HF_REPO}")
print("\n🎉 All done!")