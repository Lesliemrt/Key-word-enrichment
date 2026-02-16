
from rulebased import species_extraction_simple
from src import taxonerd

RANDOM_STATE = 42

API_KEY="1nCRukeZXLfBrLYxFpnoVhSYcIXLvTnq"

CSV_PATH = 'data/Data_cleaned.csv'
EXTENDED_CSV_PATH = 'data/Data_extended.csv'
MODELS = {
    'rulebased',
    'taxonerd',
    'scibert',
    'rulebased-scibert',
    'rulebased-taxonerd',
    'mistral'
}

scibert_model_path = "allenai/scibert_scivocab_uncased"


