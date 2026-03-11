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
├── environment.yml
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
This list is defined in config.py.

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
| Rulebased-biodivbert | 0.87 | 0.82 | 0.84 | 88.8 (21 lines) | 15.3 |
| Rulebased-scibert | 0.94 | 0.88 | 0.91 | 66.4 (21 lines) | 11.4 |

## Methodology

### Data preprocessing steps

The data is loaded from *Data_extended.csv* : EXTENDED_CSV_PATH defined in config.py.
If the model is a model that needs to be re-train, the function *train_test_split* is used with a test size of 0.3.

### Taxon extraction
For the retrained models, weights are saved in folders named checkpoints_{model_name}.
Then, the extraction function specific to each model is applied either to the whole dataset or the test dataset.

### GBIF Linking
After running models and algorithms to detect any taxons in the datasets, we’re left with a list of potential species names. We need to make sure those are actual species names, and in this case to link those species names to their GBIF website link, which will be our scientific reference.
For the linking part, we will be only using the GBIF (Global Biodiversity Information Facility) as it’s the one tool that’s advocated by the client. 
The backbone taxonomy website hosts a library containing all known taxon. To interact with the database, we will use the python library pygbif (pip install pygbif).
Using the species module from pygbif, we’re mostly using the function name_backbone, which can return a lot of information on a name. We use this function to check if the taxon name is a real one or not, and it can account for misspellings. Effectively, the function have a fuzzy matching option, set on True by default, which can help us found a taxon reference and its GBIF link even when it is misspelled.
For instance, when checking for occurrences of “Canis lupus” (common wolf), the algo￾rithm will identify exactly identify Canis lupus as an occurrence, but it will also link Canus lupus, canis lupus or Canis lps as Canis lupus, with a confidence of 85%. However, mispspellings like Canis Lupus (with both letters capitalized) will not get matched even though written correctly. This functionality is key to help us not miss any taxon names we identify in the datasets and is useful to flag mistakes on the gbif website.
Time-wise, checking an occurrence takes less than a second, slightly faster when not using fuzzy matching, so it’s important to use it as this last step of linking help us filter the false species names found and deliver a cleanly organised dictionary of couples specie name / gbif link to use as keyword for the considered dataset.

This implementation is made with the function *gbif_validation_clean.result_csv_clean* and the results are savec in *result_gbif_validated_{model}_{actual_time}.csv*

### Evaluation methodology
To check the performance of our named entity models, we used the following metrics which are precision, recall and f1 score and the runtime. We calculated these metrics score by comparing what the model extracted with our ground truth dataset which we annotated manually.

Before the gbif linking, we evaluate entity by entity and document by document with the function *evaluator.calculate_metrics*, and we save the metrics for each document in a csv file : *species_extraction_{model}_results_{actual_time}.csv*.
The general metrics are then calculated and saved in *results/RESULTS.xlsx*.

After the gbif linking, we evaluate the same way but with the function *evaluator.calculate_metrics_global* that directly returns the total precision, total recall and total f1-score. Those metrics can also be found in *results/RESULTS.xlsx*.

To be more specific, we evaluated everything strictly entity by entity. This means the model has to extract the exact and complete scientific name to be counted as correct. For example, if the real name in the text is "Ostrea edulis", the model must extract exactly "Ostrea edulis". If it only pulls out "Ostrea", we count it as a failure (a false positive for the wrong guess, and a false negative for missing the full name).
Precision: It tells out of all the entities which the model predicted as scientific names how many were actually true. If precision is high it means the model gave very less false positives.
Recall: It means out of all the actual scientific names how many did our model actually find out. High recall means that our model missed very few taxons. F1-score: It is a balance between precision and recall. Since a model could get a high recall just by guessing that almost every word is a scientific name (which would give tons of false positives and ruin precision), we need a metric that combines both. A high F1-score means the model is doing a great overall job: it is finding most of the real taxons without making too many wrong guesses.
Runtime(or duration): It tells us how quickly our model can read the metadata and extract the scientific names. Since we have 13000 datasets, a model which has a low runtime is always a good choice to use.

## Authors

- **Baka-Junior Cedric Ble**
- **Muhammad Nusrat**
- **Leslie Murat** 
- **Romain Achard**
