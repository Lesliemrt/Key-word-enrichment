# OpenMed Species Extraction Tool

## Overview
This tool uses the OpenMed NER (Named Entity Recognition) model to automatically extract species names from scientific text descriptions and validate them against the GBIF (Global Biodiversity Information Facility) taxonomic database.

## What is OpenMed?
OpenMed is a pre-trained transformer-based model specifically designed for biomedical named entity recognition. The model used here (`OpenMed-NER-SpeciesDetect-BioMed-109M`) is specialized for detecting species names in scientific literature and biodiversity datasets.

**Model Details:**
- **Type**: Transformer-based NER model (109M parameters)
- **Training**: Trained on biomedical and biodiversity texts
- **Purpose**: Identifies species names in natural language text
- **Output**: Extracts and categorizes species mentions with confidence scores

## Features
- **Automatic Species Detection**: Uses AI to find species names in text
- **GBIF Validation**: Checks extracted species against the global taxonomic database
- **Categorization**: Sorts species into accepted, misspelled, and unrecognized categories
- **Performance Metrics**: Calculates precision, recall, and accuracy
- **Batch Processing**: Processes entire CSV files at once

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Tool
```bash
python openmed_species_extraction.py
```

### 3. Check Results
The tool will create an output file: `openmed_improved_fixed_Data_cleaned.csv`

## Input Format
Your CSV file should have:
- A text column (like "Description" or "Text") containing the text to analyze
- Optionally, a "Species" column with ground truth data for evaluation

## Output Format
The tool adds these columns to your CSV:
- **predicted_species**: Comma-separated list of extracted species
- **Taxon**: List format of extracted species
- **Accepted Names**: Species found in GBIF with links
- **Misspelled Names**: Species found after correction
- **Unrecognised Names**: Species not found in GBIF

## Performance
On the included sample dataset (45 entries):
- **Precision**: 57.1%
- **Recall**: 30.3%
- **Accuracy**: 24.7%

## Files Included
- `openmed_species_extraction.py` - Main script
- `Data_cleaned.csv` - Sample input data (45 entries)
- `openmed_improved_fixed_Data_cleaned.csv` - Sample output
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Troubleshooting
- **First run is slow**: The model downloads ~400MB on first use
- **GBIF API delays**: The tool includes rate limiting for API calls
- **Memory issues**: Close other applications if you get out-of-memory errors

## Technical Details
The tool uses a multi-step pipeline:
1. Load pre-trained OpenMed NER model
2. Extract species entities from text
3. Merge and clean entity tokens
4. Validate against GBIF taxonomic database
5. Categorize results and calculate metrics
6. Save enhanced dataset with new columns

## Citation
If you use this tool in research, please cite the OpenMed model and GBIF database appropriately.