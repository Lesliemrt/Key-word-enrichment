import torch
import pandas as pd
import requests
import time
import json
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from tqdm.auto import tqdm



HF_TOKEN = "" 
login(token=HF_TOKEN)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_FILE = "Data_extended.csv" 
OUTPUT_FILE = "results_llama3_comparison.csv"

# 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.1,
    do_sample=False,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

# ==========================================
# 2. GBIF VALIDATION LOGIC
# ==========================================
GBIF_API_URL = "https://api.gbif.org/v1/species/match"
gbif_cache = {}

def check_gbif_species(name):
    name = name.strip()
    if not name or len(name) < 3: return {'valid': False}
    if name in gbif_cache: return gbif_cache[name]
    try:
        params = {'name': name, 'verbose': False}
        response = requests.get(GBIF_API_URL, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('matchType') in ['EXACT', 'FUZZY'] and data.get('confidence', 0) > 80:
                res = {
                    'valid': True, 
                    'canonical': data.get('canonicalName', name),
                    'key': data.get('usageKey', ''),
                    'matchType': data.get('matchType')
                }
                gbif_cache[name] = res
                return res
    except: pass
    gbif_cache[name] = {'valid': False}
    return gbif_cache[name]

def calculate_row_metrics(y_pred, y_true):
    """Calculates all metrics. Returns (Acc, Prec, Rec, F1)"""
    pred = set([n.lower() for n in y_pred])
    true = set([n.lower() for n in y_true])
    
    tp = len(pred.intersection(true))
    fp = len(pred - true)
    fn = len(true - pred)
    
    # CASE: Both are empty (Correctly identified absence of species)
    if len(true) == 0 and len(pred) == 0:
        return 1.0, 1.0, 1.0, 1.0
    
    # CASE: One is empty or no matches
    acc = tp / len(true.union(pred)) if len(true.union(pred)) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return round(acc, 3), round(prec, 3), round(rec, 3), round(f1, 3)

# ==========================================
# 3. EXTRACTION
# ==========================================
def get_species_from_llm(text):
    messages = [
        {"role": "system", "content": "You are a biological expert. Extract scientific species names (Latin) from text."},
        {"role": "user", "content": f"Extract scientific names as a comma-separated list. If none, return 'None'. Text: \"{text}\"\n\nNames:"},
    ]
    try:
        outputs = pipe(messages)
        raw = outputs[0]['generated_text'].strip().replace('"', '')
        if "none" in raw.lower() or not raw: return []
        return [s.strip() for s in raw.split(',') if s.strip()]
    except: return []

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    df['text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
    
    results = []

    print(f"Processing {len(df)} rows...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Ground Truth
        gt_raw = str(row.get('Species', ''))
        gt_list = [] if gt_raw.lower() in ['nan', 'none', ''] else [s.strip() for s in gt_raw.split(',') if s.strip()]
        
        # LLM Raw Prediction
        raw_preds = get_species_from_llm(row['text'])
        
        # GBIF Validation
        val_preds = []
        accepted_dict = {}
        misspelled = []
        unrecognized = []
        
        for p in raw_preds:
            check = check_gbif_species(p)
            if check['valid']: 
                val_preds.append(check['canonical'])
                accepted_dict[check['canonical']] = f"https://www.gbif.org/species/{check['key']}"
                if check['matchType'] == 'FUZZY' or p.lower() != check['canonical'].lower():
                    misspelled.append(p)
            else:
                unrecognized.append(p)
        
        # Calculate Metrics
        r_acc, r_p, r_r, r_f1 = calculate_row_metrics(raw_preds, gt_list)
        v_acc, v_p, v_r, v_f1 = calculate_row_metrics(val_preds, gt_list)
        
        results.append({
            'Title': row['Title'],
            'Ground Truth': ", ".join(gt_list),
            'LLM Raw': ", ".join(raw_preds),
            'LLM Extracted (Validated)': ", ".join(val_preds),
            # Raw Metrics
            'Raw_Precision': r_p, 'Raw_Recall': r_r, 'Raw_Accuracy': r_acc, 'Raw_F1': r_f1,
            # Validated Metrics (Matches your template column names)
            'Precision': v_p, 'Recall': v_r, 'Accuracy': v_acc, 'F1': v_f1,
            # Validation Details
            'Accepted Names JSON': json.dumps(accepted_dict),
            'Misspellings Found': ", ".join(misspelled),
            'Unrecognized': ", ".join(unrecognized)
        })

    # Save to CSV
    out_df = pd.DataFrame(results)
    # Reorder columns to match your template preference
    final_cols = [
        'Title', 'Ground Truth', 'LLM Extracted (Validated)', 
        'Precision', 'Recall', 'Accuracy', 'F1', 
        'Raw_Precision', 'Raw_Recall', 'Raw_Accuracy', 'Raw_F1',
        'Accepted Names JSON', 'Misspellings Found', 'Unrecognized'
    ]
    out_df = out_df[final_cols]
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    # Print Final Report
    print("\n" + "="*50)
    print("FINAL PERFORMANCE REPORT")
    print("-" * 50)
    print(f"{'Metric':<15} | {'Before GBIF':<15} | {'After GBIF':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<15} | {out_df['Raw_Accuracy'].mean():<15.3f} | {out_df['Accuracy'].mean():<15.3f}")
    print(f"{'Precision':<15} | {out_df['Raw_Precision'].mean():<15.3f} | {out_df['Precision'].mean():<15.3f}")
    print(f"{'Recall':<15} | {out_df['Raw_Recall'].mean():<15.3f} | {out_df['Recall'].mean():<15.3f}")
    print(f"{'F1 Score':<15} | {out_df['Raw_F1'].mean():<15.3f} | {out_df['F1'].mean():<15.3f}")
    print("="*50)
    print(f"Results saved to: {OUTPUT_FILE}")