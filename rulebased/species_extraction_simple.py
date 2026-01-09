import pandas as pd
import re

def extract_species(text) :
    if not text :
        return []
    
    patterns = {
        'parentheses': re.compile(r'\(([A-Z][a-z]{2,}\s+[a-z]{3,}(?:\s+[a-z]{3,})?)\)'),
        'binomial': re.compile(r'\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b'),
        'genus_sp': re.compile(r'\b([A-Z][a-z]{2,})\s+sp\.?\b'),
        'author': re.compile(r'\b([A-Z][a-z]{2,})\s+([a-z]{3,})\s+[A-Z][a-z]*')
    }
    
    blacklist = {
        'Lien', 'vers', 'Ces', 'traits', 'Tome', 'premier', 'Dans', 'Pour', 
        'Avec', 'Cette', 'Tous', 'Sur', 'Par', 'Une', 'Des', 'Les',
        'The', 'And', 'For', 'With', 'From', 'This', 'That', 'These', 'Those',
        'Details', 'are', 'Morphological', 'variability', 'Antarctic', 'plant',
        'Marine', 'Ocean', 'Sea', 'River', 'Lake', 'Forest', 'Park', 'Coast', 'Bay', 'Data',
        'France', 'Europe', 'Atlantic', 'Mediterranean', 'Pacific', 'Indian', 'North', 'South',
        'Argo', 'profilers', 'Microwear', 'textures', 'Enhanced', 'Observation'
    }
    
    latin_endings = [
        'us', 'a', 'um', 'is', 'e', 'ensis', 'ense', 'icus', 'ica', 'icum',
        'alis', 'ale', 'anus', 'ana', 'anum', 'inus', 'ina', 'inum',
        'osus', 'osa', 'osum', 'eus', 'ea', 'eum', 'arius', 'aria', 'arium',
        'atus', 'ata', 'atum', 'oides', 'formis', 'forme', 'ella',
        'cola', 'phila', 'philus', 'phaga', 'phagus'
    ]
    
    negative_pattern = re.compile(r'lien vers|données|campagne|étude|projet|lors de|dans le|pour le|avec le|sur le|par le', re.IGNORECASE)
    
    found = set()
    
    for match in patterns['parentheses'].finditer(text) :
        species = match.group(1).strip()
        parts = species.split()
        if len(parts) >= 2 :
            genus, epithet = parts[0], parts[1]
            context = text[max(0, match.start()-30):match.end()+30]
            if is_valid(genus, epithet, context, blacklist, latin_endings, negative_pattern) :
                found.add(f"{genus} {epithet}")
    
    for match in patterns['author'].finditer(text) :
        genus, epithet = match.groups()[:2]
        context = text[max(0, match.start()-30):match.end()+30]
        if is_valid(genus, epithet, context, blacklist, latin_endings, negative_pattern) :
            found.add(f"{genus} {epithet}")
    
    for match in patterns['binomial'].finditer(text) :
        genus, epithet = match.groups()
        context = text[max(0, match.start()-40):match.end()+40]
        if is_valid(genus, epithet, context, blacklist, latin_endings, negative_pattern) :
            found.add(f"{genus} {epithet}")
    
    for match in patterns['genus_sp'].finditer(text) :
        genus = match.group(1)
        context = text[max(0, match.start()-30):match.end()+30]
        if (genus not in blacklist and len(genus) >= 4 and 
            genus[0].isupper() and genus[1:].islower() and 
            not negative_pattern.search(context)) :
            found.add(f"{genus} sp.")
    
    return sorted(list(found))

def is_valid(genus, epithet, context, blacklist, latin_endings, negative_pattern) :
    if len(genus) < 3 or len(epithet) < 3 :
        return False
    
    if genus in blacklist or epithet in blacklist :
        return False
    
    if negative_pattern.search(context) :
        return False
    
    if not (genus[0].isupper() and genus[1:].islower()) :
        return False
    
    if not epithet.islower() :
        return False
    
    has_latin_ending = any(epithet.endswith(end) for end in latin_endings)
    has_vowels = any(v in epithet for v in 'aeiou')
    reasonable_length = 4 <= len(epithet) <= 20
    
    return has_latin_ending or (reasonable_length and has_vowels)

def calculate_metrics(extracted, ground_truth) :
    if not extracted and not ground_truth :
        return 1.0, 1.0, 1.0
    if not extracted or not ground_truth :
        return 0.0, 0.0, 0.0
    
    correct = len(extracted.intersection(ground_truth))
    precision = correct / len(extracted)
    recall = correct / len(ground_truth)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def result_csv(csv_path) :
    csv = pd.read_csv(csv_path, encoding='utf-8-sig')
    csv["Extracted"] = ""
    csv["Ground Truth"] = ""
    csv["Precision"] = ""
    csv["Recall"] = ""
    csv["F1"] = ""
    
    length = len(csv)
    print(f"Loaded {length} records")
    
    for i in range(length) :
        text = f"{csv.at[i, 'Title']} {csv.at[i, 'Description']}"
        
        extracted = set(extract_species(text))
        
        gt_text = str(csv.at[i, 'Species'])
        if gt_text in ['nan', "couldn't find", "Couldn't find", "Pas de nom"] :
            ground_truth = set()
        else :
            ground_truth = set([s.strip() for s in gt_text.split(',') if s.strip() and len(s.split()) >= 2])
        
        precision, recall, f1 = calculate_metrics(extracted, ground_truth)
        
        csv.at[i, "Extracted"] = ', '.join(sorted(extracted)) if extracted else 'None'
        csv.at[i, "Ground Truth"] = ', '.join(sorted(ground_truth)) if ground_truth else 'None'
        csv.at[i, "Precision"] = precision
        csv.at[i, "Recall"] = recall
        csv.at[i, "F1"] = f1
    
    total_extracted = sum(len(csv.at[i, "Extracted"].split(', ')) if csv.at[i, "Extracted"] != 'None' else 0 for i in range(length))
    total_ground_truth = sum(len(csv.at[i, "Ground Truth"].split(', ')) if csv.at[i, "Ground Truth"] != 'None' else 0 for i in range(length))
    total_correct = sum(
        len(set(csv.at[i, "Extracted"].split(', ')).intersection(set(csv.at[i, "Ground Truth"].split(', ')))) 
        if csv.at[i, "Extracted"] != 'None' and csv.at[i, "Ground Truth"] != 'None' else 0 
        for i in range(length)
    )
    
    precision = total_correct / total_extracted if total_extracted > 0 else 0.0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_f1 = sum(csv.at[i, "F1"] for i in range(length)) / length if length > 0 else 0.0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Total extractions: {total_extracted}")
    print(f"Correct extractions: {total_correct}")
    print(f"False positives: {total_extracted - total_correct}")
    
    csv.to_csv("species_extraction_results.csv", index=False)

if __name__ == "__main__":
    try:
        result_csv('Data.csv')
    except:
        try:
            result_csv('osama/Data.csv')
        except Exception as e:
            print(f"Error loading data: {e}")