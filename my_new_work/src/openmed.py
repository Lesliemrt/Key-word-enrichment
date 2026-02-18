import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.utils.config import openmed_model_path

# Global model and tokenizer (loaded once)
_model = None
_tokenizer = None
_device = None

def load_model():
    """Load OpenMed model and tokenizer once"""
    global _model, _tokenizer, _device
    if _model is None:
        print("Loading OpenMed model...")
        _tokenizer = AutoTokenizer.from_pretrained(openmed_model_path, add_prefix_space=True)
        _model = AutoModelForTokenClassification.from_pretrained(
            openmed_model_path, 
            num_labels=3, 
            ignore_mismatched_sizes=True
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
            
        if label == 1:  # B-SPEC (beginning of species)
            if current_entity:
                extracted.append(tokenizer.convert_tokens_to_string(current_entity))
            current_entity = [token]
        elif label == 2:  # I-SPEC (inside species)
            if current_entity:
                current_entity.append(token)
            else:
                current_entity = [token]
        else:  # O (outside)
            if current_entity:
                extracted.append(tokenizer.convert_tokens_to_string(current_entity))
                current_entity = []
    
    if current_entity:
        extracted.append(tokenizer.convert_tokens_to_string(current_entity))

    # Clean up extracted names
    clean_list = []
    for s in extracted:
        s = s.replace(" ##", "").replace("##", "").strip()
        if len(s) > 2:
            clean_list.append(s)
    
    return list(set(clean_list))
