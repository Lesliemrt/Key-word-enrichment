import pandas as pd
import re
import time

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pygbif import species

from rulebased import species_extraction_simple
from src import evaluator, taxonerd, gbif_validation_clean
from src.utils.config import CSV_PATH, EXTENDED_CSV_PATH, MODELS, scibert_model_path, RANDOM_STATE
from src.utils.utils import clean_ground_truth
from src import scibert 

def run_pipeline(model):

    if model not in MODELS:
        raise ValueError(f"Model {model} not known, choose from : {MODELS}")
    
    # ------------- Training ----------------

    if 'scibert' in model:
        print(f'---- Start preprocessing for model {model} ----')
        df = pd.read_csv(EXTENDED_CSV_PATH, encoding='utf-8-sig')
        train_df, test_df = train_test_split(df, test_size=0.33, random_state=RANDOM_STATE)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df_preprocessed = scibert.create_labels(train_df)
        tokenizer = AutoTokenizer.from_pretrained(scibert_model_path)
        train_dataset = scibert.SpeciesDataset(
            words = train_df_preprocessed['Words'], 
            labels = train_df_preprocessed['Labels'],
            tokenizer = tokenizer,
            max_len = 128)
        print(f'---- Start training for model {model} ----')
        scibert_model = scibert.SciBertForSpecies(nb_unfreezed=6)
        scibert_model = scibert.SciBert_Extended(model = scibert_model, train_dataset = train_dataset)
        scibert_model.train(epochs = 10)
        # TODO save model

    # ------------- Extraction ---------------
    if 'scibert' in model:
        csv = test_df.copy()
    else : 
        csv = pd.read_csv(EXTENDED_CSV_PATH, encoding='utf-8-sig')
    csv["Extracted"] = ""
    csv["Ground Truth"] = ""
    csv["Precision"] = ""
    csv["Recall"] = ""
    csv["F1"] = ""
    
    length = len(csv)
    print(f"Loaded {length} records")
    print(f'---- Start extraction with model {model}----')
    start_time = time.time()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(length) :
        text = f"{csv.at[i, 'Title']} {csv.at[i, 'Description']}"
        # Extraction with chosen model
        if model == 'scibert':
            extracted = set(scibert_model.extract_species(text, tokenizer))
        elif model == 'rulebased': 
            extracted = set(species_extraction_simple.extract_species(text))
        elif model == 'taxonerd' :
            extracted = set(taxonerd.extract_species(text))
        elif model == 'rulebased-scibert':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            extracted_scibert = set(scibert_model.extract_species(text, tokenizer))
            extracted = extracted_rulebased | extracted_scibert
        elif model == 'rulebased-taxonerd':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            extracted_taxonerd = set(taxonerd.extract_species(text))
            extracted = extracted_rulebased | extracted_taxonerd

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
    start_time_gbif = time.time()

    csv = gbif_validation_clean.result_csv_clean(csv)

    # Merge accepted names and mispelled names corrected
    #TODO (to have 1 in precision)

    actual_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv.to_csv(f"results/result_gbif_validated_{model}_{actual_time}.csv", index=False)
    print(f"Results saved to result_gbif_validated_{model}_{actual_time}.csv")

    print("----- Start final evaluation ----")
    extracted = csv['Accepted Names']
    extracted_names = extracted.apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])
    extracted_links = extracted.apply(lambda x: list(x.values()) if isinstance(x, dict) else [])

    end_time_gbif = time.time()
    total_duration_gbif = end_time_gbif - start_time_gbif

    print(f"\n--- REPORT AFTER GBIF CHECKED---")
    print("Name extraction : ")
    clean_gt = csv["Ground Truth"].apply(lambda x: [s.strip() for s in str(x).split(',')] if x and str(x).lower() != 'none' else [])
    precision, recall, f1 = evaluator.calculate_metrics_global(extracted_names.to_list(), clean_gt)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    print("Links extraction : ")
    clean_gt_links = csv["Gbif link"].apply(lambda x: [s.strip() for s in str(x).split(',')] if x and str(x).lower() != 'none' else [])
    precision, recall, f1 = evaluator.calculate_metrics_global(extracted_links.to_list(), clean_gt_links)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    print("GBIF duration :", total_duration_gbif)


if __name__ == "__main__":
    try:
        run_pipeline(model = 'rulebased-taxonerd')
    except Exception as e:
        print(f"Error loading data: {e}")


