import pandas as pd
import re
import time

from rulebased import species_extraction_simple
from src import evaluator, taxonerd, gbif_validation_clean

from pygbif import species

CSV_PATH = 'data/Data.csv'
MODELS = {
    'rulebased': species_extraction_simple.extract_species,
    'taxonerd': taxonerd.extract_species
}

def clean_ground_truth(gt_text):
    if str(gt_text).lower() in ['nan', "couldn't find", "pas de nom"]:
        return set()
    return set([s.strip() for s in str(gt_text).split(',') 
                if s.strip() and len(s.split()) >= 2])

def run_pipeline(model):
    csv = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    csv["Extracted"] = ""
    csv["Ground Truth"] = ""
    csv["Precision"] = ""
    csv["Recall"] = ""
    csv["F1"] = ""

    # ------------- Extraction ---------------
    
    length = len(csv)
    print(f"Loaded {length} records")
    print('---- Start extraction ----')
    start_time = time.time()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    if model not in MODELS.keys():
        raise ValueError(f"Model {model} not known, choose from : {MODELS.keys()}")

    extract_func = MODELS[model]

    for i in range(length) :
        text = f"{csv.at[i, 'Title']} {csv.at[i, 'Description']}"
        
        # Extraction with chosen model
        extracted = set(extract_func(text))

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


    # ------------- Evaluation ---------------
    total_correct = total_tp
    total_extracted = total_tp + total_fp
    total_ground_truth = total_tp + total_fn

    precision = total_correct / total_extracted if total_extracted > 0 else 0.0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_f1 = sum(csv.at[i, "F1"] for i in range(length)) / length if length > 0 else 0.0

    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n--- REPORT ---")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Total extractions: {total_extracted}")
    print(f"Correct extractions: {total_correct}")
    print(f"False positives: {total_extracted - total_correct}")
    print(f"Duration : ", total_duration)
    
    actual_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv.to_csv(f"results/species_extraction_{model}_results_{actual_time}.csv", index=False)

    # ------------- GBIF Check ---------------
    print("----- Start GBIF Check -----")

    csv = gbif_validation_clean.result_csv_clean(csv)

    actual_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv.to_csv(f"results/result_gbif_validated_{model}_{actual_time}.csv", index=False)
    print(f"Results saved to result_gbif_validated_{model}_{actual_time}.csv")

    print("----- Start final evaluation ----")
    
    extracted = csv['Accepted Names']
    extracted_names = extracted.apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])
    extracted_links = extracted.apply(lambda x: list(x.values()) if isinstance(x, dict) else [])

    print(f"\n--- REPORT AFTER GBIF CHECKED---")
    print("Name extraction : ")
    clean_gt = csv["Ground Truth"].apply(lambda x: [s.strip() for s in str(x).split(',')] if x and str(x).lower() != 'none' else [])
    precision, recall, f1 = evaluator.calculate_metrics_global(extracted_names.to_list(), clean_gt)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    print("Links extraction : ")
    clean_gt_links = csv["Gbif link"].apply(lambda x: [s.strip() for s in str(x).split(',')] if x and str(x).lower() != 'none' else [])
    precision, recall, f1 = evaluator.calculate_metrics_global(extracted_links.to_list(), clean_gt_links)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")


if __name__ == "__main__":
    try:
        run_pipeline(model = 'rulebased')
    except Exception as e:
        print(f"Error loading data: {e}")


