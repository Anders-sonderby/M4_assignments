# M4 Assignments — Green Patent Detection Pipeline

This repository contains all code, notebooks, and results for the M4 assignments in Applied Deep Learning and AI at Aalborg University. The overall goal is to build a state-of-the-art data labeling pipeline for classifying patent claims as green technology (Y02) or not, using a combination of uncertainty sampling, LLM-assisted labeling, Multi-Agent Systems, QLoRA fine-tuning, and Human-in-the-Loop (HITL) review.

---

## HuggingFace Models & Datasets

| Resource | Link |
|---|---|
| PatentSBERTa Fine-Tuned (Assignment 2) | [Anders-sonderby/patentsbert-assignment2](https://huggingface.co/Anders-sonderby/patentsbert-finetuned) |
| PatentSBERTa Fine-Tuned (Assignment 3) | [Anders-sonderby/patentsbert-assignment3](https://huggingface.co/Anders-sonderby/patentsbert-finetune_1) |
| Mistral-7B QLoRA Adapter (Final) | [Anders-sonderby/mistral-7b-patent-qlora](https://huggingface.co/Anders-sonderby/mistral-7b-patent-qlora) |
| PatentSBERTa Fine-Tuned (Final) | [Anders-sonderby/patentsbert-final-model](https://huggingface.co/Anders-sonderby/patentsbert-final-model) |
| Green Patent Dataset | [Anders-sonderby/green-patent-dataset](https://huggingface.co/datasets/Anders-sonderby/patent-green-classification) |

---

## Repository Structure

### Shared Data Files
These files are used across multiple assignments:

| File | Description |
|---|---|
| `patents_50k_green.parquet` | Balanced 50k sample (25k green, 25k not green) — base dataset for all assignments |
| `train_silver.csv` | ~40,000 silver-labeled training examples (CPC Y02* derived) |
| `eval_silver.csv` | ~5,000 silver-labeled evaluation examples |
| `hitl_green_100.csv` | 100 high-risk claims selected via uncertainty sampling — used for HITL in all assignments |

---

### Assignment 1

| File | Description |
|---|---|
| `M4_assignment_1.ipynb` | Full Assignment 1 notebook |

---

### Assignment 2 — Baseline + Simple LLM HITL

| File | Description | Notes |
|---|---|---|
| `M4_assignment_2.ipynb` | Part A: Creates `patents_50k_green.parquet` from ("AI-Growth-Lab/patents_claims_1.5m_traim_test", split="train") | |
| `M4_Assignment_2-2.ipynb` | Part A: Generates PatentSBERTa embeddings (.pt files) | Run on Google Colab. Embedding .pt files not included in repo due to size |
| `M4_Assignment_2-3.ipynb` | Part A, B & C: Baseline classifier (frozen PatentSBERTa + Logistic Regression), Uncertainty sampling → exports `hitl_green_100.csv`. LLM + Human HITL → creates `hitl_final_labaled.csv` and `hitl_gold_100.csv` | |
| `M4_Assignment_2-4.ipynb` | Part D: Fine-tunes PatentSBERTa on Silver + Gold. Creates `train_silver.csv` and `eval_silver.csv`. Evaluation metrics reported in notebook | Run on Google Colab |
| `hitl_final_labeled.csv` | LLM suggestions + human final labels for the 100 high-risk claims (Assignment 2) | |
| `hitl_gold_100.csv` | Final gold labels from Assignment 2 HITL used for PatentSBERTa fine-tuning | |

---

### Assignment 3 — Multi-Agent System (CrewAI)

| File | Description | Notes |
|---|---|---|
| `M4-Assignment_3.ipynb` | Full Assignment 3 notebook — sets up and runs the CrewAI 3-agent debate system - Using Groq API and "groq/meta-llama/llama-4-scout-17b-16e-instruct" as model | |
| `classified_results.csv` | Output from the CrewAI MAS — label + rationale for all 100 high-risk claims | |
| `hitl_final.csv` | Final gold labels after human HITL review of the 100 claims (Assignment 3) | |
| `finetune.py` | Fine-tunes PatentSBERTa on `train_silver.csv` + `hitl_final.csv`, evaluated on `eval_silver.csv` | |
| `eval_metrics.csv` | Evaluation metrics from Assignment 3 PatentSBERTa fine-tuning | |

---

### Final Assignment — QLoRA + MAS + Targeted HITL

| File | Description | Notes |
|---|---|---|
| `M4_Assignment_4.ipynb` | Part A,B, C and D: Human HITL review of MAS output → creates `hitl_gold_final.csv` and comparative analysis of final tuned model| |
| `qlora_finetune.py` | Part C Step 1: QLoRA fine-tunes Mistral-7B-v0.1 on `train_silver.csv` (40k claims, 4-bit NF4, LoRA r=16) | Run as SLURM batch job on AAU AI-Lab |
| `qlora_finetune.sh` | SLURM batch script for `qlora_finetune.py` | |
| `qlora_eval_metrics.csv` | Eval loss and runtime metrics from QLoRA fine-tuning | |
| `mas_classify.py` | Part C Step 2: 3-agent MAS — Phi-3-mini (Advocate/Skeptic) + Mistral QLoRA (Judge) — classifies 100 high-risk claims | Run as SLURM batch job on AAU AI-Lab |
| `mas_classify.sh` | SLURM batch script for `mas_classify.py` | |
| `classified_results_for_hitl.csv` | MAS output — advocate/skeptic arguments, judge label, confidence, rationale for 100 claims | |
| `hitl_gold_final.csv` | Final gold labels after targeted HITL review (only low-confidence claims reviewed) | |
| `finetune_final.py` | Part E: Final PatentSBERTa fine-tuning on `train_silver.csv` + `hitl_gold_final.csv` | |
| `eval_metrics_final.csv` | Final evaluation metrics from PatentSBERTa fine-tuning | |
| `eval_predictions_final.csv` | Per-example predictions from the final model on `eval_silver.csv` | |

---

## Model Performance Summary

| Model Version | Training Data Source | F1 Score (eval_silver) |
|---|---|---|
| 1. Baseline | Frozen Embeddings (No Fine-tuning) | 0.780 |
| 2. Assignment 2 Model | Fine-tuned on Silver + Gold (Simple LLM) | 0.818 |
| 3. Assignment 3 Model | Fine-tuned on Silver + Gold (MAS - CrewAI) | 0.824 |
| 4. Final Model | Fine-tuned on Silver + Gold (QLoRA MAS + Targeted HITL) | 0.821 |

---

## Notes

- Notebooks marked **"Run on Google Colab"** were executed on Colab due to compute or dependency requirements. All outputs and results are saved in the repo.
- Model weights are not stored in this repository. All fine-tuned models are available on the HuggingFace Hub (see links above).
- `.pt` embedding files are not included due to file size constraints.
- All batch jobs were run on the AAU AI-Lab HPC cluster using NVIDIA L4 GPUs (24GB VRAM).
