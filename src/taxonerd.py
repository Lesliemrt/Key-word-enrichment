from taxonerd import TaxoNERD
from collections import OrderedDict

import re

taxonerd = TaxoNERD(prefer_gpu=False)
nlp = taxonerd.load("en_ner_eco_biobert") # TODO : check this line

def extract_species(text):
    pattern = r'^[A-Z][a-z]+ [a-z]+$'   # Taxon name

    doc = taxonerd.find_in_text(text)  
    
    if doc is None or len(doc) == 0:
        return "couldn't find"

    taxons = [
        t for t in doc['text']
        if isinstance(t, str) and re.match(pattern, t)
    ] 
    # Supprimer les doublons 
    unique_taxons = list(OrderedDict.fromkeys(taxons))

    return sorted(taxons)