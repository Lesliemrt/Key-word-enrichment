
import torch.nn as nn
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

#TODO datapreprocessor for scibert

model_name = "allenai/scibert_scivocab_uncased"

class SciBertForSpecies(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super(SciBertForSpecies, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Input: 768 pour SciBERT base, Output: 3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs[0] 
    
        outputs = self.classifier(sequence_output)
        
        return outputs

class SciBert_Extended():
    def __init__(self, model, train_dataset, lr=2e-5):
        self.model = model
        self.dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(self.dataloader):.4f}")


