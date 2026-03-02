import os, json, re, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# 1. Environment & Auth
HF_TOKEN = os.environ.get("HF_TOKEN")
login(token=HF_TOKEN)

# 2. BitsAndBytes Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("🚀 Starting Memory-Optimized MAS...")

# --- STEP A: Load Agent Model (Phi-3) ---
phi_id = "microsoft/Phi-3-mini-4k-instruct"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_id)
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
)
phi_pipe = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer)

torch.cuda.empty_cache()

# --- STEP B: Load Judge Model (Mistral + Adapter) ---
base_model_id = "mistralai/Mistral-7B-v0.1"
m_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
m_tokenizer.pad_token = m_tokenizer.eos_token
m_tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto", max_memory={0: "12GiB"}
)
m_model = PeftModel.from_pretrained(base_model, "Anders-sonderby/mistral-7b-patent-qlora")
judge_pipe = pipeline("text-generation", model=m_model, tokenizer=m_tokenizer)

# 3. Helper Functions
def get_phi_arg(role_msg, claim):
    prompt = f"<|system|>\n{role_msg}<|end|>\n<|user|>\nClaim: {claim[:500]}<|end|>\n<|assistant|>\n"
    out = phi_pipe(prompt, max_new_tokens=100, return_full_text=False)[0]['generated_text']
    return out.strip().replace('\n', ' ')

# 4. Processing Loop
df = pd.read_csv("hitl_green_100.csv")
results = []

for i, row in df.iterrows():
    doc_id = row['doc_id']
    claim = str(row['text'])
    print(f"[{i+1}/100] Processing {doc_id}...")

    # Agents generate context
    adv_arg = get_phi_arg("You are a Green Tech Advocate. Give 2 sentences for Y02.", claim)
    skp_arg = get_phi_arg("You are a Critical Skeptic. Give 2 sentences against Y02.", claim)

    # Mistral Instruction Prompt
    j_prompt = (
        f"<s>[INST] Task: Classify the patent claim as green tech (Y02 label 1) or not (label 0).\n\n"
        f"Claim: {claim[:600]}\n"
        f"Advocate: {adv_arg}\n"
        f"Skeptic: {skp_arg}\n\n"
        "Return ONLY a JSON object with 'label' (int), 'confidence' (float 0.0-1.0), and 'rationale' (string). [/INST] {"
    )
    
    # Generate - note we prepend the { back in after generation
    j_out_raw = judge_pipe(j_prompt, max_new_tokens=200, return_full_text=False)[0]['generated_text']
    j_out = "{" + j_out_raw

    # Robust Parsing
    try:
        # Clean string from common PDF/Patent artifacts
        clean_json = j_out.replace('\xa0', ' ').replace("'", '"')
        match = re.search(r'\{.*\}', clean_json, re.DOTALL)
        if match:
            decision = json.loads(match.group())
            label = int(decision.get('label', 0))
            conf = float(decision.get('confidence', 0.5))
            rat = decision.get('rationale', "Done.")
        else:
            raise ValueError
    except:
        label = 1 if '"label": 1' in j_out else 0
        conf = 0.5
        rat = f"Fallback: {j_out[:100]}"

    results.append({
        "doc_id": doc_id,
        "text": claim,
        "ai_label": label,
        "ai_confidence": conf,
        "ai_rationale": rat
    })

    if i % 5 == 0:
        pd.DataFrame(results).to_csv("classified_results_for_hitl.csv", index=False)

pd.DataFrame(results).to_csv("classified_results_for_hitl.csv", index=False)
print("✅ Success! CSV updated with rationales.")