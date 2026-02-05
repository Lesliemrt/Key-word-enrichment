import pandas as pd
import re
import time

from rulebased import species_extraction_simple
from src import evaluator, taxonerd, gbif_validation_clean

from pygbif import species

from src.utils.config import CSV_PATH, MODELS
from src.utils.utils import clean_ground_truth
from src.scibert import create_labels


csv = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df = create_labels(csv)
print(df)