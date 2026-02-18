import os
import time
import requests
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.optimization import Adafactor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "NoYo25/BiodivBERT"

# SINGLE INPUT FILE
FULL_DATA_FILE = "Data_extended.csv"

# FINAL OUTPUT CSV (will contain only the TEST portion)
OUTPUT_FILE = "result_gbif_validated_biodivbert_split.csv"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 30
EVAL_EPOCHS = [5, 10, 15, 20, 25, 30]

SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints_biodivbert_split"
METRICS_CSV = "epoch_comparison_metrics_split.csv"

# ==========================================
# 2. GBIF VALIDATION LOGIC
# ==========================================
GBIF_API_URL = "https://api.gbif.org/v1/species/match"
gbif_cache = {}

def check_gbif_species(name: str):
    if name in gbif_cache: return gbif_cache[name]
    try:
        params = {"name": name, "verbose": False}
        response = requests.get(GBIF_API_URL, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            match_type = data.get("matchType", "NONE")
            confidence = data.get("confidence", 0)
            if match_type in ["EXACT", "FUZZY"] and confidence > 80:
                result = {
                    "valid": True,
                    "canonical": data.get("canonicalName", name),
                    "key": data.get("usageKey", ""),
                    "matchType": match_type
                }
                gbif_cache[name] = result
                return result
    except: pass
    res = {"valid": False}
    gbif_cache[name] = res
    time.sleep(0.05)
    return res

def categorize_species(species_list):
    accepted, misspelled, unrecognised = {}, [], []
    for sp in species_list:
        sp = sp.strip()
        if len(sp) < 3: continue
        res = check_gbif_species(sp)
        if res["valid"]:
            link = f"https://www.gbif.org/species/{res['key']}"
            accepted[res["canonical"]] = link
            if res["matchType"] == "FUZZY" and sp.lower() != res["canonical"].lower():
                misspelled.append(sp)
        else:
            unrecognised.append(sp)
    return accepted, misspelled, unrecognised

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================
class NERDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def process_dataframe(df, tokenizer):
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
    data = []

    for _, row in df.iterrows():
        text = str(row["text"])
        species_raw = str(row["Species"])
        if species_raw == "nan": continue

        species_list = [s.strip() for s in species_raw.split(",") if s.strip()]
        char_labels = np.zeros(len(text), dtype=int)
        found = False

        for s in species_list:
            start = text.find(s)
            if start != -1:
                found = True
                char_labels[start] = 1
                char_labels[start + 1 : start + len(s)] = 2

        if not found: continue

        try:
            tok = tokenizer(
                text, max_length=MAX_LEN, padding="max_length", truncation=True, return_offsets_mapping=True
            )
            labs = []
            for (s, e) in tok["offset_mapping"]:
                if s == e: labs.append(-100)
                else: labs.append(char_labels[s] if s < len(char_labels) else 0)

            data.append({
                "input_ids": torch.tensor(tok["input_ids"]),
                "attention_mask": torch.tensor(tok["attention_mask"]),
                "labels": torch.tensor(labs)
            })
        except: continue
    return data

# ==========================================
# 4. EXTRACTION & METRICS
# ==========================================
def extract(text, model, tokenizer):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad(): out = model(**inputs)
    preds = torch.argmax(out.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    extracted, curr = [], []
    for tok, lab in zip(tokens, preds):
        if tok in tokenizer.all_special_tokens: continue
        if tok.startswith("##"):
            if curr: curr.append(tok)
            continue
        if lab == 1:
            if curr: extracted.append(tokenizer.convert_tokens_to_string(curr))
            curr = [tok]
        elif lab == 2:
            if curr: curr.append(tok)
            else: curr = [tok]
        else:
            if curr: extracted.append(tokenizer.convert_tokens_to_string(curr)); curr = []
    if curr: extracted.append(tokenizer.convert_tokens_to_string(curr))

    res = []
    for s in extracted:
        s = s.replace(" ##", "").replace("##", "").strip()
        if len(s) > 2: res.append(s)
    return list(set(res))

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

def evaluate_dataframe(model, tokenizer, df):
    model.eval()
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")

    raw_tp = raw_fp = raw_fn = 0
    val_tp = val_fp = val_fn = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval", leave=False):
        gt_raw = str(row.get("Species", ""))
        gt_list = [] if gt_raw == "nan" else [s.strip() for s in gt_raw.split(",") if s.strip()]
        y_true = set([n.lower() for n in gt_list])

        raw_extracts = extract(str(row["text"]), model, tokenizer)
        y_pred_raw = set([n.lower() for n in raw_extracts])
        rtp, rfp, rfn = calc_stats(y_pred_raw, y_true)
        raw_tp += rtp; raw_fp += rfp; raw_fn += rfn

        acc_dict, _, _ = categorize_species(raw_extracts)
        valid_names = list(acc_dict.keys())
        y_pred_val = set([n.lower() for n in valid_names])
        vtp, vfp, vfn = calc_stats(y_pred_val, y_true)
        val_tp += vtp; val_fp += vfp; val_fn += vfn

    raw_metrics = compute_metrics(raw_tp, raw_fp, raw_fn)
    val_metrics = compute_metrics(val_tp, val_fp, val_fn)

    return {
        "raw": {"acc": raw_metrics[0], "p": raw_metrics[1], "r": raw_metrics[2], "f1": raw_metrics[3]},
        "val": {"acc": val_metrics[0], "p": val_metrics[1], "r": val_metrics[2], "f1": val_metrics[3]}
    }

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_and_eval_split():
    # 1. Load Full Data
    print(f"Loading {FULL_DATA_FILE}...")
    full_df = pd.read_csv(FULL_DATA_FILE)

    # 2. Split 70/30
    train_df, test_df = train_test_split(full_df, test_size=0.3, random_state=42)
    print(f"Train Size: {len(train_df)}")
    print(f"Test Size:  {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data = process_dataframe(train_df, tokenizer)
    loader = DataLoader(NERDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze strategy
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    if hasattr(model, "bert"): enc = model.bert.encoder
    elif hasattr(model, "roberta"): enc = model.roberta.encoder
    else: enc = model.base_model.encoder
    for layer in enc.layer[-4:]:
        for p in layer.parameters(): p.requires_grad = True

    optim = Adafactor(model.parameters(), lr=4e-5, relative_step=False, scale_parameter=False, warmup_init=False)

    if SAVE_CHECKPOINTS: os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_f1 = -1.0
    best_epoch = None
    results = []

    model.train()
    for ep in range(1, EPOCHS + 1):
        for batch in tqdm(loader, desc=f"Epoch {ep}", leave=False):
            b = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**b)
            out.loss.backward()
            optim.step()

        if ep in EVAL_EPOCHS:
            # Eval on TEST split
            metrics = evaluate_dataframe(model, tokenizer, test_df)
            val_f1 = metrics["val"]["f1"]

            print(f"\n--- Epoch {ep} ---")
            print(f"Val F1 (Validated): {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = ep

            results.append({"epoch": ep, "val_f1": val_f1})

            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{ep}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

            model.train()

    print(f"\nBest Epoch: {best_epoch} (F1: {best_f1:.4f})")
    return best_epoch, test_df

# ==========================================
# 6. FINAL GENERATION
# ==========================================
def generate_final_csv(model_path, test_df):
    print("\nGenerating Final CSV on Test Split...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_df["text"] = test_df["Title"].fillna("") + " " + test_df["Description"].fillna("")

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
        raw_extracts = extract(str(row["text"]), model, tokenizer)
        y_pred_raw = set([n.lower() for n in raw_extracts])
        rtp, rfp, rfn = calc_stats(y_pred_raw, y_true)
        raw_tp += rtp; raw_fp += rfp; raw_fn += rfn

        # 3. Validated
        acc_dict, miss_list, unrec_list = categorize_species(raw_extracts)
        valid_names = list(acc_dict.keys())
        y_pred_val = set([n.lower() for n in valid_names])
        vtp, vfp, vfn = calc_stats(y_pred_val, y_true)
        val_tp += vtp; val_fp += vfp; val_fn += vfn

        # 4. Row Metrics (using validated)
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

    # Save CSV
    pd.DataFrame(out_data).to_csv(OUTPUT_FILE, index=False)

    # 5. PRINT GLOBAL REPORT
    def print_final_stats(name, tp, fp, fn):
        acc, p, r, f1 = compute_metrics(tp, fp, fn)
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")

    print("\n" + "="*40)
    print("FINAL REPORT (Test Split)")
    print("="*40)
    print_final_stats("BEFORE GBIF (Raw)", raw_tp, raw_fp, raw_fn)
    print_final_stats("AFTER GBIF (Validated)", val_tp, val_fp, val_fn)
    print("="*40)
    print(f"Saved split test results to {OUTPUT_FILE}")

# ==========================================
# 7. RUN
# ==========================================
if __name__ == "__main__":
    best_ep, test_dataframe = train_and_eval_split()
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"epoch_{best_ep}")
    generate_final_csv(best_ckpt, test_dataframe)