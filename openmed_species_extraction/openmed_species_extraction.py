import os
import pandas as pd
import re
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
import time
from typing import Dict, List, Tuple, Set

MODEL_NAME = "OpenMed/OpenMed-NER-SpeciesDetect-BioMed-109M"
MAX_CHARS = 500
INPUT_CSV = "Data_cleaned.csv"
GT_COLUMN = "Species"
OUTPUT_PREFIX = "openmed_improved_"
NOISE_PREFIXES = {
    "red deer",
    "roe deer",
    "chamois",
    "mouflon",
    "les",
    "mer",
    "don",
    "cre",
    "sol",
    "ou",
    "le",
    "la",
}


def load_ner_pipeline():
    print("Loading OpenMed model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    print("Model loaded!")
    return ner


def run_ner(text, ner):
    if not isinstance(text, str):
        return []

    # Try multiple approaches with the same model
    results = []

    # Approach 1: Full text (up to MAX_CHARS)
    full_results = ner(text[:MAX_CHARS])
    if isinstance(full_results, list) and len(full_results) > 0:
        # Handle return_all_scores=True case
        if isinstance(full_results[0], list):
            results.extend(full_results[0])
        else:
            results.extend(full_results)

    # Approach 2: Process in smaller chunks for better detection
    chunk_size = 200
    for i in range(0, min(len(text), MAX_CHARS), chunk_size):
        chunk = text[i : i + chunk_size]
        if len(chunk.strip()) > 10:  # Only process meaningful chunks
            chunk_results = ner(chunk)
            if isinstance(chunk_results, list) and len(chunk_results) > 0:
                if isinstance(chunk_results[0], list):
                    results.extend(chunk_results[0])
                else:
                    results.extend(chunk_results)

    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for item in results:
        # Create a unique key for each entity
        key = (item.get("word", ""), item.get("start", 0), item.get("end", 0))
        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    return unique_results


def post_process_species_name(name):
    if not name:
        return ""

    # Remove noise prefixes using string operations (no regex)
    for p in NOISE_PREFIXES:
        if name.startswith(p + " "):
            name = name[len(p) :].strip()

    # Clean common tokenization artifacts using string operations
    name = name.replace("##", "")  # Remove BERT subword tokens
    name = name.replace("[CLS]", "").replace("[SEP]", "")  # Remove special tokens

    return name.lower().strip()


def is_valid_species(name):
    if not name or len(name) < 4:
        return False
    if len(name.split()) >= 2:
        return True
    if len(name) >= 8:
        return True
    return False


def merge_species_entities(entities):
    species = set()
    i = 0

    while i < len(entities):
        e = entities[i]
        if "SPECIES" not in e["entity_group"].upper():
            i += 1
            continue

        name = e["word"]
        end = e["end"]
        j = i + 1

        # Look ahead to merge tokens that belong together
        while j < len(entities):
            n = entities[j]
            if "SPECIES" in n["entity_group"].upper() and abs(n["start"] - end) <= 10:  # Increased tolerance
                token = n["word"]
                # Handle BERT subword tokens
                if token.startswith("##"):
                    name += token[2:]  # Remove ## and append directly
                else:
                    name += " " + token
                end = n["end"]
                j += 1
            else:
                break

        # Clean and validate the merged name
        cleaned_name = post_process_species_name(name.lower().strip())

        # More lenient validation for species names
        if is_valid_species_relaxed(cleaned_name):
            species.add(cleaned_name)

        i = j

    return list(species)


def is_valid_species_relaxed(name):
    """More lenient species validation"""
    if not name or len(name) < 3:
        return False

    # Split into words
    words = name.split()

    # Accept binomial nomenclature (2 words)
    if len(words) == 2:
        genus, species_epithet = words
        if len(genus) >= 3 and len(species_epithet) >= 3:
            return True

    # Accept single genus names if long enough
    if len(words) == 1 and len(name) >= 6:
        return True

    # Accept longer names that might include additional info
    if len(words) > 2:
        # Check if first two words look like genus + species
        if len(words[0]) >= 3 and len(words[1]) >= 3:
            return True

    return False


def extract_species(text, ner):
    entities = run_ner(text, ner)
    names = merge_species_entities(entities)
    return names


def check_gbif_species(species_name: str) -> Tuple[bool, str]:
    """Check if species exists in GBIF and return status and URL"""
    try:
        # Clean the species name - capitalize first letter of each word
        clean_name = species_name.strip()

        # Convert to proper scientific name format (Genus species)
        parts = clean_name.split()
        if len(parts) >= 2:
            clean_name = f"{parts[0].capitalize()} {parts[1].lower()}"
        else:
            clean_name = clean_name.capitalize()

        print(f"Checking GBIF for: '{clean_name}'")

        # GBIF species match API
        url = f"https://api.gbif.org/v1/species/match"
        params = {"name": clean_name}

        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()
            print(
                f"GBIF response for '{clean_name}': matchType={data.get('matchType')}, confidence={data.get('confidence')}"
            )

            # Check if we got a good match
            if data.get("matchType") in ["EXACT", "FUZZY"] and data.get("usageKey"):
                # Also check confidence score
                confidence = data.get("confidence", 0)
                if confidence >= 90:  # Only accept high confidence matches
                    gbif_url = f"https://www.gbif.org/species/{data['usageKey']}"
                    print(f"✓ Found match: {gbif_url}")
                    return True, gbif_url
                else:
                    print(f"✗ Low confidence match ({confidence})")
            else:
                print(f"✗ No good match found")
        else:
            print(f"✗ GBIF API error: {response.status_code}")

        return False, ""

    except Exception as e:
        print(f"Error checking GBIF for {species_name}: {e}")
        return False, ""


def categorize_species(
    species_list: List[str],
) -> Tuple[Dict[str, str], List[List[str]], List[str]]:
    """Categorize species into accepted, misspelled, and unrecognised

    Returns: (accepted_dict, misspelled_list, unrecognised_list)
    """
    accepted_names = {}
    misspelled_names = []
    unrecognised_names = []

    for species in species_list:
        if not species or len(species.strip()) < 3:
            continue

        print(f"\nProcessing species: '{species}'")

        # Add small delay to avoid rate limiting
        time.sleep(0.5)  # Increased delay

        is_valid, gbif_url = check_gbif_species(species)

        if is_valid:
            accepted_names[species] = f"GBIF taxonomy page - {gbif_url}"
            print(f"✓ Added to accepted: {species}")
        else:
            # Try some common variations/corrections
            variations = []

            # Try with different capitalizations
            parts = species.split()
            if len(parts) >= 2:
                variations.extend(
                    [
                        f"{parts[0].capitalize()} {parts[1].lower()}",
                        f"{parts[0].upper()} {parts[1].lower()}",
                        species.title(),
                        species.upper(),
                    ]
                )

            found_match = False
            for variation in variations:
                if variation != species:  # Don't recheck the same format
                    time.sleep(0.3)
                    is_valid_var, gbif_url_var = check_gbif_species(variation)
                    if is_valid_var:
                        # This is a misspelled name that we corrected
                        misspelled_names.append(
                            [
                                f"'{species}'",
                                variation,
                                f"GBIF taxonomy page - {gbif_url_var}",
                            ]
                        )
                        print(f"✓ Found correction: '{species}' -> '{variation}'")
                        found_match = True
                        break

            if not found_match:
                unrecognised_names.append(f"'{species}'")
                print(f"✗ Added to unrecognised: {species}")

    return accepted_names, misspelled_names, unrecognised_names


def normalize_species_name(name):
    name = str(name).lower().strip()

    # Remove common French articles using string operations
    if name.startswith("le "):
        name = name[3:]
    elif name.startswith("la "):
        name = name[3:]
    elif name.startswith("les "):
        name = name[4:]
    elif name.startswith("du "):
        name = name[3:]
    elif name.startswith("des "):
        name = name[4:]

    # Normalize multiple spaces to single space
    while "  " in name:
        name = name.replace("  ", " ")

    return name.strip()


def parse_species_list(text):
    if pd.isna(text):
        return set()

    # Clean brackets and quotes using string operations
    text = str(text)
    text = text.replace("[", "").replace("]", "").replace("'", "")

    species_set = set()
    for s in text.split(","):
        s = s.strip()
        if len(s) > 3:
            normalized = normalize_species_name(s)
            species_set.add(normalized)

    return species_set


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def evaluate_predictions(df):
    tp = fp = fn = 0

    for _, row in df.iterrows():
        gt = parse_species_list(row[GT_COLUMN])
        pr = parse_species_list(row["predicted_species"])

        matched = set()
        for p in pr:
            for g in gt:
                if p == g or similarity(p, g) >= 0.8:
                    matched.add((p, g))

        tp += len(matched)
        fp += len(pr) - len({m[0] for m in matched})
        fn += len(gt) - len({m[1] for m in matched})

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0

    print("\n--- Evaluation ---")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")


def process_csv(csv_path, ner):
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

    text_cols = [
        c for c in df.columns if "description" in c.lower() or "text" in c.lower()
    ]
    if not text_cols:
        raise ValueError("No text column found")

    text_col = text_cols[0]
    print(f"Using text column: {text_col}")

    # Extract species using NER
    print("Extracting species...")
    species_lists = []
    for idx, text in enumerate(df[text_col]):
        if idx % 10 == 0:
            print(f"Processing row {idx}/{len(df)}")
        species_list = extract_species(text, ner)
        species_lists.append(species_list)

    df["predicted_species"] = [", ".join(species) for species in species_lists]

    # Add the new columns in the same format as result_checked.csv
    print("Categorizing species with GBIF validation...")
    taxon_column = []
    accepted_names_column = []
    misspelled_names_column = []
    unrecognised_names_column = []

    for idx, species_list in enumerate(species_lists):
        if idx % 5 == 0:
            print(f"Validating species for row {idx}/{len(species_lists)}")

        # Taxon column: list of extracted species
        taxon_column.append(str(species_list))

        # Categorize species
        accepted, misspelled, unrecognised = categorize_species(species_list)
        accepted_names_column.append(str(accepted))
        misspelled_names_column.append(str(misspelled))
        unrecognised_names_column.append(str(unrecognised))

    # Add the new columns
    df["Taxon"] = taxon_column
    df["Accepted Names"] = accepted_names_column
    df["Misspelled Names"] = misspelled_names_column
    df["Unrecognised Names"] = unrecognised_names_column

    if GT_COLUMN in df.columns:
        evaluate_predictions(df)

    out = OUTPUT_PREFIX + "fixed_" + os.path.basename(csv_path)
    df.to_csv(out, index=False)
    print(f"\nOutput saved to {out}")
    print(f"Added columns: Taxon, Accepted Names, Misspelled Names, Unrecognised Names")


def main():
    print("Loading OpenMed model...")
    ner = load_ner_pipeline()
    print("Model loaded")
    process_csv(INPUT_CSV, ner)


if __name__ == "__main__":
    main()