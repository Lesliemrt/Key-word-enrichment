import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.utils.config import biodivbert_model_path

# Global model and tokenizer (loaded once)
_model = None
_tokenizer = None
_device = None

def load_model():
    """Load BiodivBERT model and tokenizer once"""
    global _model, _tokenizer, _device
    if _model is None:
        print("Loading BiodivBERT model...")
        _tokenizer = AutoTokenizer.from_pretrained(biodivbert_model_path)
        _model = AutoModelForTokenClassification.from_pretrained(biodivbert_model_path, num_labels=3)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        print(f"BiodivBERT loaded on {_device}")
    return _model, _tokenizer, _device

def extract_species(text):
    """
    Extract species names from text using BiodivBERT
    
    Args:
        text: Input text string
        
    Returns:
        List of extracted species names
    """
    model, tokenizer, device = load_model()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    extracted = []
    current = []
    
    for tok, label in zip(tokens, preds):
        if tok in tokenizer.all_special_tokens:
            continue
            
        if tok.startswith("##"):
            if current:
                current.append(tok)
            continue
            
        if label == 1:  # B-SPEC (beginning of species)
            if current:
                extracted.append(tokenizer.convert_tokens_to_string(current))
            current = [tok]
        elif label == 2:  # I-SPEC (inside species)
            if current:
                current.append(tok)
            else:
                current = [tok]
        else:  # O (outside)
            if current:
                extracted.append(tokenizer.convert_tokens_to_string(current))
                current = []
    
    if current:
        extracted.append(tokenizer.convert_tokens_to_string(current))

    # Clean up extracted names
    result = []
    for s in extracted:
        s = s.replace(" ##", "").replace("##", "").strip()
        if len(s) > 2:
            result.append(s)
    
    return list(set(result))
