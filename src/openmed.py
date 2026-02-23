import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.optimization import Adafactor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.utils.config import openmed_model_path, openmed_checkpoint_dir, EXTENDED_CSV_PATH, RANDOM_STATE


MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 30  
EVAL_EPOCHS = [5, 10, 15, 20, 25, 30] 


_model = None
_tokenizer = None
_device = None


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def process_dataframe(df, tokenizer):
    """Process dataframe for training"""
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
    data = []

    for _, row in df.iterrows():
        text = str(row["text"])
        species_raw = str(row["Species"])
        if species_raw == "nan":
            continue

        species_list = [s.strip() for s in species_raw.split(",") if s.strip()]
        char_labels = np.zeros(len(text), dtype=int)
        found = False

        for s in species_list:
            start = text.find(s)
            if start != -1:
                found = True
                char_labels[start] = 1
                char_labels[start + 1 : start + len(s)] = 2

        if not found:
            continue

        try:
            tok = tokenizer(
                text, max_length=MAX_LEN, padding="max_length", 
                truncation=True, return_offsets_mapping=True
            )
            labs = []
            for (s, e) in tok["offset_mapping"]:
                if s == e:
                    labs.append(-100)
                else:
                    labs.append(char_labels[s] if s < len(char_labels) else 0)

            data.append({
                "input_ids": torch.tensor(tok["input_ids"]),
                "attention_mask": torch.tensor(tok["attention_mask"]),
                "labels": torch.tensor(labs)
            })
        except:
            continue
    return data


def calc_stats(y_pred, y_true):
    """Calculate TP, FP, FN"""
    tp = len(y_pred.intersection(y_true))
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    """Calculate accuracy, precision, recall, F1"""
    total = tp + fp + fn
    acc = tp / total if total > 0 else 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return acc, p, r, f1

def evaluate_on_test(model, tokenizer, test_df):
    """Evaluate model on test set and return F1 score"""
    model.eval()
    test_df["text"] = test_df["Title"].fillna("") + " " + test_df["Description"].fillna("")
    
    tp = fp = fn = 0
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating", leave=False):
        gt_raw = str(row.get("Species", ""))
        gt_list = [] if gt_raw == "nan" else [s.strip() for s in gt_raw.split(",") if s.strip()]
        y_true = set([n.lower() for n in gt_list])
        
        text = str(row["text"])
        extracts = extract_from_model(text, model, tokenizer)
        y_pred = set([n.lower() for n in extracts])
        
        rtp, rfp, rfn = calc_stats(y_pred, y_true)
        tp += rtp
        fp += rfp
        fn += rfn
    
    acc, p, r, f1 = compute_metrics(tp, fp, fn)
    return f1


def train_openmed(train_df, test_df, epochs=EPOCHS, checkpoint_dir=None):
    """
    Train OpenMed model with checkpoint saving and best epoch selection
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe for evaluation
        epochs: Number of training epochs
        checkpoint_dir: Custom checkpoint directory (optional)
        
    Returns:
        Best model, tokenizer, and best epoch number
    """
    print(f"---- Training OpenMed for {epochs} epochs ----")
    
   
    ckpt_dir = checkpoint_dir if checkpoint_dir else openmed_checkpoint_dir
    
    tokenizer = AutoTokenizer.from_pretrained(openmed_model_path, add_prefix_space=True)
    train_data = process_dataframe(train_df, tokenizer)
    loader = DataLoader(NERDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(
        openmed_model_path, num_labels=3, ignore_mismatched_sizes=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "bert"):
        enc = model.bert.encoder
    else:
        enc = model.base_model.encoder
    for layer in enc.layer[-4:]:
        for p in layer.parameters():
            p.requires_grad = True

    optim = Adafactor(model.parameters(), lr=4e-5, relative_step=False, 
                     scale_parameter=False, warmup_init=False)

    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_f1 = -1.0
    best_epoch = None
    best_model_state = None

    
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0
        for batch in tqdm(loader, desc=f"OpenMed Epoch {ep}/{epochs}", leave=False):
            b = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**b)
            out.loss.backward()
            optim.step()
            total_loss += out.loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"OpenMed Epoch {ep}/{epochs} | Loss: {avg_loss:.4f}")
        
        
        if ep in EVAL_EPOCHS:
            val_f1 = evaluate_on_test(model, tokenizer, test_df)
            print(f"  → Validation F1: {val_f1:.4f}")
            
            
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{ep}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  → Checkpoint saved to {ckpt_path}")
            
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = ep
                best_model_state = model.state_dict().copy()
                print(f"  →  NEW BEST MODEL! (F1: {best_f1:.4f})")
            
            model.train()  

    print(f"\nOpenMed training complete ")
    print(f"Best Epoch: {best_epoch} | Best F1: {best_f1:.4f}")
    
   
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, tokenizer, best_epoch

def extract_from_model(text, model, tokenizer):
    """Extract species from text using a given model"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())

    extracted = []
    current_entity = []
    
    for token, label in zip(tokens, predictions):
        if token in tokenizer.all_special_tokens:
            continue
            
        if token.startswith("##"):
            if current_entity:
                current_entity.append(token)
            continue
            
        if label == 1:  
            if current_entity:
                extracted.append(tokenizer.convert_tokens_to_string(current_entity))
            current_entity = [token]
        elif label == 2:  
            if current_entity:
                current_entity.append(token)
            else:
                current_entity = [token]
        else:  
            if current_entity:
                extracted.append(tokenizer.convert_tokens_to_string(current_entity))
                current_entity = []
    
    if current_entity:
        extracted.append(tokenizer.convert_tokens_to_string(current_entity))

 
    clean_list = []
    for s in extracted:
        s = s.replace(" ##", "").replace("##", "").strip()
        if len(s) > 2:
            clean_list.append(s)
    
    return list(set(clean_list))

def load_model():
    """Load OpenMed model from checkpoint or base model"""
    global _model, _tokenizer, _device
    if _model is None:
        
        if os.path.exists(openmed_checkpoint_dir):
            epochs = [int(d.split('_')[1]) for d in os.listdir(openmed_checkpoint_dir) 
                     if d.startswith('epoch_') and os.path.isdir(os.path.join(openmed_checkpoint_dir, d))]
            if epochs:
                best_epoch = max(epochs)
                checkpoint_path = os.path.join(openmed_checkpoint_dir, f"epoch_{best_epoch}")
                print(f"Loading OpenMed from checkpoint: {checkpoint_path}")
                _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, add_prefix_space=True)
                _model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
            else:
                print("No checkpoints found, loading base OpenMed...")
                _tokenizer = AutoTokenizer.from_pretrained(openmed_model_path, add_prefix_space=True)
                _model = AutoModelForTokenClassification.from_pretrained(
                    openmed_model_path, num_labels=3, ignore_mismatched_sizes=True
                )
        else:
            print("No checkpoints found, loading base OpenMed...")
            _tokenizer = AutoTokenizer.from_pretrained(openmed_model_path, add_prefix_space=True)
            _model = AutoModelForTokenClassification.from_pretrained(
                openmed_model_path, num_labels=3, ignore_mismatched_sizes=True
            )
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        print(f"OpenMed loaded on {_device}")
    return _model, _tokenizer, _device

def extract_species(text):
    """
    Extract species names from text using OpenMed
    
    Args:
        text: Input text string
        
    Returns:
        List of extracted species names
    """
    model, tokenizer, device = load_model()
    return extract_from_model(text, model, tokenizer)
