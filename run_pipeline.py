import pandas as pd
import re
import time
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pygbif import species

from rulebased import species_extraction_simple
from src import evaluator, taxonerd, gbif_validation_clean, biodivbert, openmed, llamainstruct
from src.utils.config import CSV_PATH, EXTENDED_CSV_PATH, MODELS, scibert_model_path, openmed_model_path, biodivbert_model_path, llama_model_path, RANDOM_STATE
from src.utils.utils import clean_ground_truth
from src import scibert 

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
# Pour garantir le déterminisme sur GPU (attention, peut ralentir l'entraînement)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run_pipeline(model):

    if model not in MODELS:
        raise ValueError(f"Model {model} not known, choose from : {MODELS}")
    
   

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
        scibert_model.train(epochs = 30)
        

    if 'biodivbert' in model:
        print(f'---- Start training BiodivBERT ----')
        df = pd.read_csv(EXTENDED_CSV_PATH, encoding='utf-8-sig')
        train_df, test_df = train_test_split(df, test_size=0.33, random_state=RANDOM_STATE)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        

        if model == 'rulebased-biodivbert':
            checkpoint_dir = "checkpoints_rulebased_biodivbert"
        else:
            checkpoint_dir = "checkpoints_biodivbert_split"
        
        biodivbert_model, biodivbert_tokenizer, best_epoch = biodivbert.train_biodivbert(
            train_df, test_df, epochs=30, checkpoint_dir=checkpoint_dir
        )
        print(f"Using best BiodivBERT model from epoch {best_epoch}")

    if 'openmed' in model:
        print(f'---- Start training OpenMed ----')
        df = pd.read_csv(EXTENDED_CSV_PATH, encoding='utf-8-sig')
        train_df, test_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        

        if model == 'rulebased-openmed':
            checkpoint_dir = "checkpoints_rulebased_openmed"
        else:
            checkpoint_dir = "checkpoints_openmed_split"
        
        openmed_model, openmed_tokenizer, best_epoch = openmed.train_openmed(
            train_df, test_df, epochs=30, checkpoint_dir=checkpoint_dir
        )
        print(f"Using best OpenMed model from epoch {best_epoch}")


    if 'scibert' in model:
        csv = test_df.copy()
    elif 'biodivbert' in model:
        csv = test_df.copy()
    elif 'openmed' in model:
        csv = test_df.copy()
    else:
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

    
    # if model == 'mistral':
    #     print("---- Start Mistral extraction (batching) ----")
    #     texts = [f"{csv.at[i, 'Title']} {csv.at[i, 'Description']}" for i in range(length)]
    #     mistral_results = mistral.batch_extract(texts, batch_size=20, sleep_between=0.5)

    for i in range(length) :
        text = f"{csv.at[i, 'Title']} {csv.at[i, 'Description']}"
       
        
        if model == 'scibert':
            extracted = set(scibert_model.extract_species(text, tokenizer))
        elif model == 'rulebased': 
            extracted = set(species_extraction_simple.extract_species(text))
        elif model == 'taxonerd' :
            extracted = set(taxonerd.extract_species(text))
        elif model == 'biodivbert':
            
            if 'biodivbert_model' in locals():
                extracted = set(biodivbert.extract_from_model(text, biodivbert_model, biodivbert_tokenizer))
            else:
                extracted = set(biodivbert.extract_species(text))
        elif model == 'openmed':
            
            if 'openmed_model' in locals():
                extracted = set(openmed.extract_from_model(text, openmed_model, openmed_tokenizer))
            else:
                extracted = set(openmed.extract_species(text))
        elif model == 'llamainstruct':
            extracted = set(llamainstruct.extract_species(text))
        elif model == 'rulebased-scibert':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            extracted_scibert = set(scibert_model.extract_species(text, tokenizer))
            extracted = extracted_rulebased | extracted_scibert
        elif model == 'rulebased-taxonerd':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            extracted_taxonerd = set(taxonerd.extract_species(text))
            extracted = extracted_rulebased | extracted_taxonerd
        elif model == 'rulebased-biodivbert':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            if 'biodivbert_model' in locals():
                extracted_biodivbert = set(biodivbert.extract_from_model(text, biodivbert_model, biodivbert_tokenizer))
            else:
                extracted_biodivbert = set(biodivbert.extract_species(text))
            extracted = extracted_rulebased | extracted_biodivbert
        elif model == 'rulebased-openmed':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            if 'openmed_model' in locals():
                extracted_openmed = set(openmed.extract_from_model(text, openmed_model, openmed_tokenizer))
            else:
                extracted_openmed = set(openmed.extract_species(text))
            extracted = extracted_rulebased | extracted_openmed
        elif model == 'rulebased-llamainstruct':
            extracted_rulebased = set(species_extraction_simple.extract_species(text))
            extracted_llamainstruct = set(llamainstruct.extract_species(text))
            extracted = extracted_rulebased | extracted_llamainstruct
        # elif model == 'mistral':
        #     raw_output = mistral_results[i]
        #     extracted = set(raw_output) if raw_output else set()

        
        ground_truth = clean_ground_truth(csv.at[i, 'Species'])
        
        
        precision, recall, f1 = evaluator.calculate_metrics(extracted, ground_truth)

        total_tp += len(extracted & ground_truth)
        total_fp += len(extracted - ground_truth)
        total_fn += len(ground_truth - extracted)
        
        
        csv.at[i, "Extracted"] = ', '.join(sorted(extracted)) if extracted else 'None'
        csv.at[i, "Ground Truth"] = ', '.join(sorted(ground_truth)) if ground_truth else 'None'
        csv.at[i, "Precision"] = precision
        csv.at[i, "Recall"] = recall
        csv.at[i, "F1"] = f1

    results = {}
   
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
    
    results["Before GBIF Check"] = {'Precision' : precision, 'Recall' : recall, 'F1-score' : f1, 'Duration' : total_duration}
    
    import os
    results_dir = f"results/{model}"
    os.makedirs(results_dir, exist_ok=True)
    
    actual_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv.to_csv(f"{results_dir}/species_extraction_{model}_results_{actual_time}.csv", index=False)

   
    print(" Start GBIF Check ")
    start_time_gbif = time.time()

    csv = gbif_validation_clean.result_csv_clean(csv)

   


    actual_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv.to_csv(f"{results_dir}/result_gbif_validated_{model}_{actual_time}.csv", index=False)
    print(f"Results saved to {results_dir}/result_gbif_validated_{model}_{actual_time}.csv")

   
    extracted = csv['Accepted Names']
    extracted_names = extracted.apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])
    extracted_links = extracted.apply(lambda x: list(x.values()) if isinstance(x, dict) else [])

    end_time_gbif = time.time()
    total_duration_gbif = end_time_gbif - start_time_gbif

    print(f"\n REPORT AFTER GBIF CHECKED")
    print("Name extraction : ")
    clean_gt = csv["Ground Truth"].apply(lambda x: [s.strip() for s in str(x).split(',')] if x and str(x).lower() != 'none' else [])
    precision, recall, f1 = evaluator.calculate_metrics_global(extracted_names.to_list(), clean_gt)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    results["After GBIF Check - Name extraction"] = {'Precision' : precision, 'Recall' : recall, 'F1-score' : f1, 'Duration' : total_duration_gbif}
    print("Links extraction : ")
    clean_gt_links = csv["Gbif link"].apply(lambda x: [s.strip() for s in str(x).split(',')] if x and str(x).lower() != 'none' else [])
    precision, recall, f1 = evaluator.calculate_metrics_global(extracted_links.to_list(), clean_gt_links)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
    print("GBIF duration :", total_duration_gbif)

    results["After GBIF Check - Links"] = {'Precision' : precision, 'Recall' : recall, 'F1-score' : f1, 'Duration' : total_duration_gbif}
    df_results = pd.DataFrame(results)

    file_path = f"results/RESULTS.xlsx"
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine="openpyxl", mode='a', if_sheet_exists="replace") as writer:
            df_results.to_excel(writer, sheet_name=f"{model}")
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode='w') as writer:
            df_results.to_excel(writer, sheet_name=f"{model}")



if __name__ == "__main__":
    try:
        run_pipeline(model = 'scibert')
    except Exception as e:
        print(f"Error loading data: {e}")



