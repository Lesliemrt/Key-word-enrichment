
from rulebased import species_extraction_simple
from src import taxonerd

RANDOM_STATE = 42

API_KEY="1nCRukeZXLfBrLYxFpnoVhSYcIXLvTnq"
HF_TOKEN = ""

CSV_PATH = 'data/Data_cleaned.csv'
EXTENDED_CSV_PATH = 'data/Data_extended.csv'
MODELS = {
    'rulebased',
    'taxonerd',
    'scibert',
    'biodivbert',
    'openmed',
    'llamainstruct',
    'rulebased-scibert',
    'rulebased-taxonerd',
    'rulebased-biodivbert',
    'rulebased-openmed',
    'rulebased-llamainstruct',
    'mistral'
}

scibert_model_path = "allenai/scibert_scivocab_uncased"
biodivbert_model_path = "NoYo25/BiodivBERT"
openmed_model_path = "OpenMed/OpenMed-NER-SpeciesDetect-BioMed-109M"
llamainstruct_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"


