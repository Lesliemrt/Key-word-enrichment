# Key-word-enrichment

## Overview

The classification of living organisms or the scientific classification of species is an important
discipline of an institution such as the National Museum of Natural History. It is based on
systematics and taxonomy. Taxonomy is a branch of the natural sciences that studies the
diversity of the living world. This activity consists of describing and circumscribing living
organisms in terms of species and organizing them into hierarchical categories called taxa.
The objective of systematics is to unambiguously name a species by means of a scientific
name that follows a nomenclature : It is composed of a Latin binomial or trinomial which
consists of a genus (Genus), a specific epithet that represents the species, and an infraspecific
epithet that represents the subspecies. To this must be added information concerning the
author(s) of the taxon, as well as a date. This scientific name uniquely identifies a species.
Here are few example of scientific names in zoology :
- Pentastiridius badiensis van Stalle, 1986
- Medmassa semiaurantiaca Simon, 1910
- Cabirnalia nausicaa Boyko & van der Meij, 2018
- Blastomussa vivida Benzoni, Arrigoni & Hoeksema, 2014
The PNDB website hosts more than 13000 datasets that contains scientific names and
that biology researchers use regularly. The objective is to come up with a way to identify
any scientific specie names in a given dataset and use it as a keyword for this dataset. In
this way the PNDB website would be easier to use and it would be easier to find research
on a given species.

## Project Structure

```bash
.
├── data/
│   └── Data_extended.csv              # Dataset with ground truth
├── notebooks/                         
├── results/
│   └── RESULTS.xlsx                   # Results of different models
├── rulebased/
├── src/                               
│   ├── utils/                         
│   │   ├── config.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── biodivbert.py                 
│   ├── evaluator.py 
│   ├── gbif_validation_clean.py
│   ├── llamainstruct.py
│   ├── mistral.py
│   ├── openmed.py
│   ├── scibert.py
│   └── taxonerd.py
├── .gitignore
├── environment.yml                    # Configuration de l'environnement Conda
├── llamaonly.ipynb
├── README.md
├── result_gbif_validated.csv
└── run_pipeline.py 


```

## Installation

```bash
# Install dependencies and package
conda env create -f environment.yml
conda activate env_procom_clean
```

## Usage

To run the pipeline, the following command can be used :

```bash
# Run basic pipeline
python run_pipeline.py --model <model_name>
```

The model can be chosed from the following options : 
    'rulebased',
    'taxonerd',
    'scibert',
    'biodivbert',
    'openmed',
    'llamainstruct',
    'rulebased-scibert',
    'rulebased-taxonerd',
    'rulebased-biodivbert',
    'rulebased-openmed',
    'rulebased-llamainstruct',
    'mistral'

## Model Performance

| Model | Precision | Recall | F1-score | Duration (s) / Test Size | Expected Dur. All (h) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| Rulebased | 0.98 | 0.72 | 0.83 | 199.69 (70 lines) | 10.3 |
| Taxonerd | 1.00 | 0.63 | 0.77 | 164.13 (70 lines) | 8.5 |
| Scibert | 0.94 | 0.88 | 0.90 | 67.11 (21 lines) | 11.5 |
| Biodivbert | 0.94 | 0.82 | 0.87 | 53.1 (21 lines) | 9.2 |
| Openmed | 1.00 | 0.47 | 0.64 | 67 (21 lines) | 11.4 |
| Meta-llama | 0.95 | 0.95 | 0.95 | 353 (70 lines) | 18.2 |
| Rulebased-taxonerd | 0.98 | 0.83 | 0.90 | 391.75 (70 lines) | 20.2 |
| Rulebased-biodivbert | 0.87 | 0.82 | 0.84 | 168.75 (21 lines) | 29.0 |


## Methodology

### Data preprocessing steps

#### 1. Data loading
The data is load from *Data_extended.csv* : EXTENDED_CSV_PATH defined in config.py.


### Model selection


### Evaluation methodology

## Authors

- **Baka-Junior Cedric Ble**
- **Muhammad Nusrat**
- **Leslie Murat** 
- **Romain Achard**
