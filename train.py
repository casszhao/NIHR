from typing import Any

import torch
import pandas as pd
import numpy as np
import re
import nltk
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

df = pd.read_csv('training_data_sentences.csv')


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label_list = df['label']
le.fit(label_list)
encoded_label = le.transform(label_list)

df['encoded'] = encoded_label
training_set = df[['encoded', 'sentences']]


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(training_set['sentences'], training_set['encoded'], test_size=.2)
test_texts, val_texts, test_labels, val_labels = train_test_split(val_texts, val_labels, test_size=.5)


model_name = 'emilyalsentzer/Bio_ClinicalBERT'
model = AutoModelForSequenceClassification.from_pretrained(model_name) # num_labels= len(df['label'].unique()
tokenizer = AutoTokenizer.from_pretrained(model_name)


train_encodings = tokenizer(str(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(str(val_texts), truncation=True, padding=True)
test_encodings = tokenizer(str(test_texts), truncation=True, padding=True)



class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(1):
    for batch in train_loader:
        print(batch)

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()