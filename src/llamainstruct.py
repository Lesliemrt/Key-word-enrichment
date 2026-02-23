import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login
from src.utils.config import llama_model_path, HF_TOKEN


_pipe = None
_tokenizer = None

def load_model():
    """Load Llama 3.1 8B Instruct model with 4-bit quantization"""
    global _pipe, _tokenizer
    if _pipe is None:
        print("Loading Llama 3.1 8B Instruct model...")
        
        
        login(token=HF_TOKEN)
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            _tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                llama_model_path,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
        else:
            
            print("CUDA not available, loading on CPU (this may be slow)...")
            _tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                llama_model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
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
        
        print(f"Llama 3.1 8B Instruct loaded on {device}")
    return _pipe, _tokenizer

def extract_species(text):
    """
    Extract species names from text using Llama 3.1 8B Instruct
    
    Args:
        text: Input text string
        
    Returns:
        List of extracted species names
    """
    pipe, tokenizer = load_model()
    
    messages = [
        {"role": "system", "content": "You are a biological expert. Extract scientific species names (Latin) from text."},
        {"role": "user", "content": f"Extract scientific names as a comma-separated list. If none, return 'None'. Text: \"{text}\"\n\nNames:"},
    ]
    
    try:
        outputs = pipe(messages)
        raw = outputs[0]['generated_text'].strip().replace('"', '')
        
        if "none" in raw.lower() or not raw:
            return []
        
        # Parse comma-separated list
        species_list = [s.strip() for s in raw.split(',') if s.strip()]
        return species_list
    except Exception as e:
        print(f"Error in Llama extraction: {e}")
        return []
