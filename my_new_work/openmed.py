import pandas as pd
import numpy as np
import torch
import requests
import time
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.optimization import Adafactor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "OpenMed/OpenMed-NER-SpeciesDetect-BioMed-109M"

# SINGLE FILE INPUT
FULL_DATA_FILE = "Data_extended.csv"
OUTPUT_FILE = "results_openmed_split_extended.csv"
METRICS_LOG_FILE = "training_log_metrics_openmed.csv"

# Training Settings
MAX_LEN = 256
BATCH_SIZE = 16
TOTAL_EPOCHS = 30
EVAL_SCHEDULE = [5, 10, 15, 20, 25, 30]
CHECKPOINT_DIR = "checkpoints_openmed_split"

# GBIF Settings
GBIF_API_URL = "https://api.gbif.org/v1/species/match"
gbif_cache = {}

# ==========================================
# 2. GBIF VALIDATION FUNCTION
# ==========================================
def check_gbif_species(name):
    if name in gbif_cache: return gbif_cache[name]
    try:
        params = {'name': name, 'verbose': False}
        response = requests.get(GBIF_API_URL, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            match_type = data.get('matchType', 'NONE')
            confidence = data.get('confidence', 0)
            if match_type in ['EXACT', 'FUZZY'] and confidence > 80:
                result = {
                    'valid': True,
                    'canonical': data.get('canonicalName', name),
                    'key': data.get('usageKey', ''),
                    'matchType': match_type
                }
                gbif_cache[name] = result
                return result
    except: pass
    res = {'valid': False}
    gbif_cache[name] = res
    time.sleep(0.05)
    return res

def categorize_species(species_list):
    accepted = {}
    misspelled = []
    unrecognised = []
    for sp in species_list:
        sp = sp.strip()
        if len(sp) < 3: continue
        res = check_gbif_species(sp)
        if res['valid']:
            link = f"https://www.gbif.org/species/{res['key']}"
            accepted[res['canonical']] = link
            if res['matchType'] == 'FUZZY' and sp.lower() != res['canonical'].lower():
                 misspelled.append(sp)
        else:
            unrecognised.append(sp)
    return accepted, misspelled, unrecognised

# ==========================================
# 3. DATA PREPARATION
# ==========================================
def load_and_label_data(source, tokenizer):
    if isinstance(source, str): df = pd.read_csv(source)
    else: df = source.copy()

    df['text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
    data = []

    for idx, row in df.iterrows():
        text = str(row['text'])
        species_raw = str(row['Species'])
        if species_raw == 'nan': continue

        species_list = [s.strip() for s in species_raw.split(',') if s.strip()]
        char_labels = np.zeros(len(text), dtype=int)
        found_any = False

        for species in species_list:
            start_idx = text.find(species)
            if start_idx != -1:
                found_any = True
                end_idx = start_idx + len(species)
                char_labels[start_idx] = 1
                char_labels[start_idx+1:end_idx] = 2

        if not found_any: continue

        try:
            tokenized = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_offsets_mapping=True)
            labels = []
            for (start, end) in tokenized['offset_mapping']:
                if start == end: labels.append(-100)
                else: labels.append(char_labels[start])

            data.append({
                'input_ids': torch.tensor(tokenized['input_ids']),
                'attention_mask': torch.tensor(tokenized['attention_mask']),
                'labels': torch.tensor(labels)
            })
        except: continue
    return data

class NERDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==========================================
# 4. INFERENCE & SCORING UTILS
# ==========================================
def extract_species(text, model, tokenizer):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad(): outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())

    extracted, current_ent = [], []
    for token, label in zip(tokens, predictions):
        if token in tokenizer.all_special_tokens: continue
        if token.startswith("##"):
            if current_ent: current_ent.append(token)
            continue

        if label == 1:
            if current_ent: extracted.append(tokenizer.convert_tokens_to_string(current_ent))
            current_ent = [token]
        elif label == 2:
            if current_ent: current_ent.append(token)
            else: current_ent = [token]
        else:
            if current_ent:
                extracted.append(tokenizer.convert_tokens_to_string(current_ent))
                current_ent = []
    if current_ent: extracted.append(tokenizer.convert_tokens_to_string(current_ent))

    clean_list = []
    for s in extracted:
        s = s.replace(" ##", "").replace("##", "").strip()
        if len(s) > 2: clean_list.append(s)
    return list(set(clean_list))

def calc_stats(y_pred, y_true):
    tp = len(y_pred.intersection(y_true))
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    total = tp + fp + fn
    acc = tp / total if total > 0 else 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return acc, p, r, f1

# ==========================================
# 5. EVALUATION LOOP (Check Epoch Quality)
# ==========================================
def evaluate_epoch(model, tokenizer, test_df):
    model.eval()
    df_eval = test_df.copy()
    df_eval['text'] = df_eval['Title'].fillna('') + " " + df_eval['Description'].fillna('')

    raw_tp = raw_fp = raw_fn = 0
    val_tp = val_fp = val_fn = 0

    # Eval Loop
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Eval", leave=False):
        gt_raw = str(row.get("Species", ""))
        gt_list = [] if gt_raw == "nan" else [s.strip() for s in gt_raw.split(",") if s.strip()]
        y_true = set([n.lower() for n in gt_list])

        # Raw
        raw_extracts = extract_species(str(row["text"]), model, tokenizer)
        y_pred_raw = set([n.lower() for n in raw_extracts])
        rtp, rfp, rfn = calc_stats(y_pred_raw, y_true)
        raw_tp += rtp; raw_fp += rfp; raw_fn += rfn

        # Validated
        acc_dict, _, _ = categorize_species(raw_extracts)
        valid_names = list(acc_dict.keys())
        y_pred_val = set([n.lower() for n in valid_names])
        vtp, vfp, vfn = calc_stats(y_pred_val, y_true)
        val_tp += vtp; val_fp += vfp; val_fn += vfn

    return {
        "raw": compute_metrics(raw_tp, raw_fp, raw_fn),
        "val": compute_metrics(val_tp, val_fp, val_fn)
    }

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Setup
    if os.path.exists(CHECKPOINT_DIR): shutil.rmtree(CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    # 1. Load & Split
    print(f"Loading {FULL_DATA_FILE}...")
    full_df = pd.read_csv(FULL_DATA_FILE)
    train_df, test_df = train_test_split(full_df, test_size=0.30, random_state=42)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Add 'text' column to test_df for final generation
    test_df['text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')

    # 2. Prepare Data
    processed_data = load_and_label_data(train_df, tokenizer)
    train_loader = DataLoader(NERDataset(processed_data), batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Unfreeze
    for param in model.parameters(): param.requires_grad = False
    for param in model.classifier.parameters(): param.requires_grad = True
    if hasattr(model, 'bert'): encoder = model.bert.encoder
    else: encoder = model.base_model.encoder
    for layer in encoder.layer[-4:]:
        for param in layer.parameters(): param.requires_grad = True

    optimizer = Adafactor(model.parameters(), lr=4e-5, relative_step=False, scale_parameter=False, warmup_init=False)

    # 4. TRAINING LOOP
    best_f1 = -1
    best_epoch = 0
    metrics_log = []

    print(f"Starting Training for {TOTAL_EPOCHS} Epochs...")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # A. Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            b = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**b)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()

        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

        # B. Evaluate & Save (If scheduled)
        if epoch in EVAL_SCHEDULE:
            res = evaluate_epoch(model, tokenizer, test_df)
            val_f1 = res['val'][3] # F1 index

            print(f"-> Validated F1: {val_f1:.4f}")

            # Log
            metrics_log.append({
                'Epoch': epoch,
                'Raw_F1': res['raw'][3], 'Val_F1': res['val'][3]
            })

            # Save Checkpoint
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

            # Track Best
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                print(f"🌟 NEW BEST MODEL! (Epoch {epoch})")

    # 5. FINAL GENERATION (Using Best Model)
    print(f"\nLoading Best Checkpoint from Epoch {best_epoch}...")
    best_model = AutoModelForTokenClassification.from_pretrained(os.path.join(CHECKPOINT_DIR, f"epoch_{best_epoch}"))
    best_model.to(device)
    best_model.eval()

    print("Generating Final CSV & Report...")

    # Global Counters
    raw_tp = raw_fp = raw_fn = 0
    val_tp = val_fp = val_fn = 0

    out_data = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # 1. Truth
        gt_raw = str(row.get("Species", ""))
        gt_list = [] if gt_raw == "nan" else [s.strip() for s in gt_raw.split(",") if s.strip()]
        y_true = set([n.lower() for n in gt_list])

        # 2. Raw
        raw_extracts = extract_species(str(row["text"]), best_model, tokenizer)
        y_pred_raw = set([n.lower() for n in raw_extracts])
        rtp, rfp, rfn = calc_stats(y_pred_raw, y_true)
        raw_tp += rtp; raw_fp += rfp; raw_fn += rfn

        # 3. Validated
        acc_dict, miss_list, unrec_list = categorize_species(raw_extracts)
        valid_names = list(acc_dict.keys())
        y_pred_val = set([n.lower() for n in valid_names])
        vtp, vfp, vfn = calc_stats(y_pred_val, y_true)
        val_tp += vtp; val_fp += vfp; val_fn += vfn

        # 4. Row Metrics
        denom = vtp + vfp + vfn
        acc = vtp / denom if denom > 0 else (1.0 if not y_pred_val and not y_true else 0.0)
        p = vtp / (vtp + vfp) if (vtp + vfp) > 0 else (1.0 if not y_pred_val else 0.0)
        r = vtp / (vtp + vfn) if (vtp + vfn) > 0 else (1.0 if not y_true else 0.0)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        row_dict = row.to_dict()
        row_dict.update({
            "Extracted": ", ".join(sorted(valid_names)),
            "Ground Truth": ", ".join(sorted(gt_list)),
            "Precision": p, "Recall": r, "F1": f1, "Accuracy": acc,
            "Accepted Names": str(acc_dict),
            "Misspelled Names": str(miss_list),
            "Unrecognised Names": str(unrec_list)
        })
        out_data.append(row_dict)

    pd.DataFrame(out_data).to_csv(OUTPUT_FILE, index=False)

    # 6. PRINT FINAL REPORT
    def print_final_stats(name, tp, fp, fn):
        acc, p, r, f1 = compute_metrics(tp, fp, fn)
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")

    print("\n" + "="*40)
    print(f"OPENMED FINAL REPORT (Best Epoch {best_epoch})")
    print("="*40)
    print_final_stats("BEFORE GBIF (Raw)", raw_tp, raw_fp, raw_fn)
    print_final_stats("AFTER GBIF (Validated)", val_tp, val_fp, val_fn)
    print("="*40)
    print(f"Results saved to {OUTPUT_FILE}")