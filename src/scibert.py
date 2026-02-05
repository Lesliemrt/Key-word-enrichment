import re
import torch.nn as nn
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.utils.utils import clean_ground_truth
from src.utils.config import scibert_model_path

def create_labels(csv):
    '''
    Create labels for each word : 0 if it's not a species name, 1 for the first word of a species name, 2 for the other

    Args :
        df : Dataframe with column Title, Description, Species (species name in the text)
    Returns :
        Same df with column 'Words' = title+description separated by words and 'Labels' list of label for each word
    '''
    df = csv.copy()
    df['Words'] = None
    df['Text'] = None
    df['Labels'] = None
    for idx in range(len(df)):
        
        text = f"{df.loc[idx, 'Title']} {df.loc[idx, 'Description']}"
        df.loc[idx, 'Text'] = text

        words = re.findall(r"[\w']+|[.,!?;()]", text)
        df.at[idx, 'Words'] = words

        labels = [0] * len(words)
        # ground_truth = df.loc[idx,'Species']
        ground_truth = clean_ground_truth(df.at[idx, 'Species'])
        if ground_truth:
            for species in ground_truth:
                species_tokens = re.findall(r"[\w']+|[.,!?;()]", species)
                i = 0
                while i <= len(words) - 2:
                    segment = words[i : i + 2]
                    if [w.lower() for w in segment] == [s.lower() for s in species_tokens]:
                        labels[i] = 1  # B-SPEC
                        labels[i + 1] = 2  # I-SPEC
                        i += 2
                    else:
                        i += 1
        df.at[idx, 'Labels'] = labels
    return df

def align_labels_with_tokens(labels, word_ids):
    '''
    Align labels with tokens

    arg:
    labels : list of labels for word index (0 if the word is not part of a species name, 1 is it is the first word, 2 if it's the second)
    word_ids : 
    '''
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word: #beginning of a new word
            new_labels.append(labels[word_id])
            current_word = word_id
        else:
            if labels[word_id] in [1, 2]:
                new_labels.append(2)
            else:
                new_labels.append(0)
    return new_labels

import torch
from torch.utils.data import Dataset

class SpeciesDataset(Dataset):
    def __init__(self, words, labels, tokenizer, max_len=128):
        self.words = words   # Text cutted into words (list of list of words)
        self.labels = labels 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        text = self.words[item]
        word_labels = self.labels[item]

        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        aligned_labels = align_labels_with_tokens(word_labels, word_ids)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


class SciBertForSpecies(nn.Module):
    def __init__(self, num_labels=3, nb_unfreezed = 3):
        super(SciBertForSpecies, self).__init__()
        self.bert = AutoModel.from_pretrained(scibert_model_path)
        self.num_labels = num_labels
        # Input: 768 pour SciBERT base, Output: 3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
        for param in self.bert.parameters():
            param.requires_grad = False

        # Defreeze layers
        for layer in self.bert.encoder.layer[-nb_unfreezed:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] 
        outputs = self.classifier(sequence_output)
        return outputs

class SciBert_Extended():
    def __init__(self, model, train_dataset, weights = torch.tensor([1.0, 3.0, 1.0]), lr=2e-5):
        self.model = model
        self.dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.compute_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.compute_loss(outputs.view(-1, self.model.num_labels), labels.view(-1))
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(self.dataloader):.4f}")

    def extract_species(self, text, tokenizer, max_len=128):
        self.model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                            padding='max_length', max_length=max_len)
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        preds = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()
        
        # From label to words
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        word_ids = inputs.word_ids()

        species_found = []
        current_entity = ""

        for i, label in enumerate(preds):
            if word_ids[i] is None:
                continue
            token = tokens[i]
            if label == 1:
                if current_entity:
                    species_found.append(current_entity.strip())
                current_entity = token.replace("##", "")
                print("label=1 current species", current_entity)
            elif label == 2:
                if current_entity:
                    if token.startswith("##"):
                        current_entity += token.replace("##", "")
                    else:
                        current_entity += " " + token
                else :
                    current_entity = token.replace("##", "")
                print("label = 2 current species", current_entity)
            else:
                if current_entity:
                    species_found.append(current_entity.strip())
                    print(" current species", current_entity)
                    current_entity = ""

        print(species_found)
        print(type(species_found))

        return species_found



