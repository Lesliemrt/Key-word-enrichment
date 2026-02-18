import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from src.utils.config import llamainstruct_model_path, HF_TOKEN

# Global model and pipeline (loaded once)
_pipe = None
_tokenizer = None

def load_model():
    """Load LlamaInstruct model and pipeline once"""
    global _pipe, _tokenizer
    if _pipe is None:
        print("Loading LlamaInstruct model (this may take a while)...")
        
        # Login to HuggingFace
        login(token=HF_TOKEN)
        
        # 4-bit Quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        _tokenizer = AutoTokenizer.from_pretrained(llamainstruct_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            llamainstruct_model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
        
        _pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=_tokenizer,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
            pad_token_id=_tokenizer.eos_token_id
        )
        
        print("LlamaInstruct loaded successfully")
    return _pipe, _tokenizer

def extract_species(text):
    """
    Extract species names from text using LlamaInstruct
    
    Args:
        text: Input text string
        
    Returns:
        List of extracted species names
    """
    pipe, tokenizer = load_model()
    
    messages = [
        {
            "role": "system", 
            "content": "You are a biological expert. Extract scientific species names (Latin) from text."
        },
        {
            "role": "user", 
            "content": f"Extract scientific names as a comma-separated list. If none, return 'None'. Text: \"{text}\"\n\nNames:"
        },
    ]
    
    try:
        outputs = pipe(messages)
        raw = outputs[0]['generated_text'].strip().replace('"', '')
        
        # Handle "None" or empty responses
        if "none" in raw.lower() or not raw:
            return []
        
        # Parse comma-separated list
        species_list = [s.strip() for s in raw.split(',') if s.strip()]
        return species_list
    except Exception as e:
        print(f"Error in LlamaInstruct extraction: {e}")
        return []
