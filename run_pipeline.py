import pandas as pd
import re
import time

from rulebased import species_extraction_simple, gbif_validation_clean
from src import evaluator
from src import gbif_check
from src import taxonerd

from pygbif import species

CSV_PATH = 'data/Data.csv'

def clean_ground_truth(gt_text):
    if str(gt_text).lower() in ['nan', "couldn't find", "pas de nom"]:
        return set()
    return set([s.strip() for s in str(gt_text).split(',') 
                if s.strip() and len(s.split()) >= 2])

def print_report(precision, recall, f1, avg_f1, total_extracted, total_correct):
    print(f"\n--- REPORT ---")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Total extractions: {total_extracted}")
    print(f"Correct extractions: {total_correct}")
    print(f"False positives: {total_extracted - total_correct}")

def run_pipeline(model):

    csv = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    csv["Extracted"] = ""
    csv["Ground Truth"] = ""
    csv["Precision"] = ""
    csv["Recall"] = ""
    csv["F1"] = ""
    
    length = len(csv)
    print(f"Loaded {length} records")
    print('---- Start extraction ----')
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for i in range(length) :
        text = f"{csv.at[i, 'Title']} {csv.at[i, 'Description']}"
        
        # Extraction with choosed model
        #TODO : choisir fonction avant pour juste avoir à faire extracted = extract()
        if model == 'rulebased':
            extracted = set(species_extraction_simple.extract_species(text))
        elif model == 'taxonerd':
            extracted = set(taxonerd.extract_species(text)) #TODO

        # Clean ground truth
        ground_truth = clean_ground_truth(csv.at[i, 'Species'])
        
        # Evaluation line by line
        precision, recall, f1 = evaluator.calculate_metrics(extracted, ground_truth)

        total_tp += len(extracted & ground_truth)
        total_fp += len(extracted - ground_truth)
        total_fn += len(ground_truth - extracted)
        
        # Save results
        csv.at[i, "Extracted"] = ', '.join(sorted(extracted)) if extracted else 'None'
        csv.at[i, "Ground Truth"] = ', '.join(sorted(ground_truth)) if ground_truth else 'None'
        csv.at[i, "Precision"] = precision
        csv.at[i, "Recall"] = recall
        csv.at[i, "F1"] = f1


    # Global evaluation
    total_correct = total_tp
    total_extracted = total_tp + total_fp
    total_ground_truth = total_tp + total_fn

    precision = total_correct / total_extracted if total_extracted > 0 else 0.0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(csv['F1'])
    avg_f1 = sum(csv.at[i, "F1"] for i in range(length)) / length if length > 0 else 0.0
    
    print_report(precision, recall, f1, avg_f1, total_extracted, total_correct)
    
    # csv.to_csv("species_extraction_results.csv", index=False)

    # GBIF Check
    print("----- Start GBIF Check -----")

    csv = gbif_validation_clean.result_csv(csv)

    csv.to_csv("result_gbif_validated.csv", index=False)
    print("Results saved to result_gbif_validated.csv")
    
if __name__ == "__main__":
    try:
        run_pipeline(model = 'rulebased')
    except Exception as e:
        print(f"Error loading data: {e}")


