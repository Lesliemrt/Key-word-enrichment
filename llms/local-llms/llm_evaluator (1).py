#!/usr/bin/env python3
import pandas as pd
import json
import re
import time
import requests
from typing import List, Dict
import numpy as np

class LLMEvaluator:
    def __init__(self, models: List[str] = None):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.models = models or [
            'codellama:7b',
            'deepseek-coder-v2:16b',
            'deepseek-coder:6.7b',
            'llama3.1:8b',
            'mistral',
            'mistral:7b'
        ]
        
    def create_prompt(self, text: str) -> str:
        return f"""Extract all scientific names (binomial nomenclature) from this biodiversity text.

Rules:
- Only binomial names (Genus species)
- No common names or author citations
- Return as JSON list: ["Species name1", "Species name2"]
- If none found, return: []

Text: "{text[:1000]}"

JSON response:"""

    def query_model(self, model_name: str, prompt: str) -> List[str]:
        try:
            response = requests.post(self.ollama_url, json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9}
            }, timeout=60)
            
            if response.status_code == 200:
                result = response.json()["response"].strip()
                return self.parse_response(result)
        except:
            pass
        return []

    def parse_response(self, response: str) -> List[str]:
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                species_list = json.loads(json_match.group())
                if isinstance(species_list, list):
                    return [s for s in species_list if isinstance(s, str) and self.is_valid_binomial(s)]
            except:
                pass
        return []

    def is_valid_binomial(self, name: str) -> bool:
        if not name or len(name.split()) != 2:
            return False
        genus, species = name.split()
        if len(genus) < 3 or len(species) < 3:
            return False
        if not (genus[0].isupper() and species[0].islower()):
            return False
        false_positives = {
            'Les données', 'Le tableau', 'La zone', 'Ces données', 'Cette étude',
            'Lien vers', 'Atlas thématique', 'JSON response', 'Species name',
            'Text analysis', 'Data extraction', 'Scientific names'
        }
        return name not in false_positives

    def clean_ground_truth(self, species_text: str) -> List[str]:
        if pd.isna(species_text):
            return []
        cleaned = re.sub(r'\s+', ' ', str(species_text).strip())
        species_list = []
        for species in cleaned.split(','):
            species = species.strip()
            if species and self.is_valid_binomial(species):
                species_list.append(species)
        return species_list

    def calculate_metrics(self, ground_truth: List[str], predicted: List[str]) -> Dict:
        gt_set = set(ground_truth)
        pred_set = set(predicted)
        
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        perfect_match = gt_set == pred_set
        partial_match = tp > 0
        
        return {
            'perfect_match': perfect_match,
            'partial_match': partial_match,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def evaluate(self, csv_file: str, output_file: str = 'llm_evaluation_results.csv'):
        df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
        df['Ground_Truth'] = df['Species'].apply(self.clean_ground_truth)
        
        for model_name in self.models:
            print(f"Evaluating {model_name}...")
            
            predictions = []
            matches = []
            precisions = []
            recalls = []
            times = []
            
            for idx, row in df.iterrows():
                ground_truth = self.clean_ground_truth(row.get('Species', ''))
                text = f"{row.get('Title', '')} {row.get('Description', '')}"
                prompt = self.create_prompt(text)
                
                start_time = time.time()
                extracted = self.query_model(model_name, prompt)
                processing_time = time.time() - start_time
                
                metrics = self.calculate_metrics(ground_truth, extracted)
                
                predictions.append(extracted)
                matches.append('Perfect' if metrics['perfect_match'] else 'Partial' if metrics['partial_match'] else 'No Match')
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                times.append(processing_time)
            
            clean_name = model_name.replace(':', '_').replace('.', '_')
            df[f'{clean_name}_Predictions'] = predictions
            df[f'{clean_name}_Match'] = matches
            df[f'{clean_name}_Precision'] = precisions
            df[f'{clean_name}_Recall'] = recalls
            df[f'{clean_name}_Time'] = times
        
        df.to_csv(output_file, index=False)
        self.create_summary(df, output_file.replace('.csv', '_summary.csv'))
        print(f"Results saved to {output_file}")

    def create_summary(self, df: pd.DataFrame, summary_file: str):
        summary_data = []
        
        for model_name in self.models:
            clean_name = model_name.replace(':', '_').replace('.', '_')
            
            if f'{clean_name}_Match' in df.columns:
                matches = df[f'{clean_name}_Match']
                precisions = df[f'{clean_name}_Precision']
                recalls = df[f'{clean_name}_Recall']
                times = df[f'{clean_name}_Time']
                
                perfect = sum(1 for m in matches if m == 'Perfect')
                partial = sum(1 for m in matches if m == 'Partial')
                no_match = sum(1 for m in matches if m == 'No Match')
                
                summary_data.append({
                    'Model': model_name,
                    'Perfect_Matches': perfect,
                    'Partial_Matches': partial,
                    'No_Matches': no_match,
                    'Perfect_Match_Rate': f"{perfect/len(df)*100:.1f}%",
                    'Avg_Precision': f"{np.mean(precisions):.3f}",
                    'Avg_Recall': f"{np.mean(recalls):.3f}",
                    'Avg_Time_Seconds': f"{np.mean(times):.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary:\n{summary_df.to_string(index=False)}")

if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.evaluate('Data.csv')
