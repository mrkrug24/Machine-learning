import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW

LABELS = [
    'AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 
    'EVENT', 'FACILITY', 'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 
    'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PENALTY', 
    'PERCENT', 'PERSON', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', 
    'TIME', 'WORK_OF_ART'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
task_3_entities = ['O'] + [f"{p}-{t}" for t in LABELS for p in ['B', 'I']]
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModelForTokenClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=len(task_3_entities)).to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['input_ids'], self.data[idx]['labels']

def ann_processing(str):
    items = str.split('\t')
    if ';' in items[1]: return None
    start, end, entity = list(map(int, items[1].split()[1:3])) + [items[1].split()[0]]
    return start, end, entity

def data_processing(txt, ann):
    with open(txt, 'r', encoding='utf-8') as t, open(ann, 'r', encoding='utf-8') as a:
        txt_text = t.read()
        ann_text = [ann_processing(line) for line in a if line.strip().startswith('T')]
    return txt_text, sorted(filter(None, ann_text), key=lambda x: x[0])

def load(dir, data = []):
    for file in filter(lambda f: f.endswith('.txt'), os.listdir(dir)):
        ann_path = os.path.join(dir, file[:-4] + '.ann')
        if os.path.exists(ann_path):
            data.append(data_processing(os.path.join(dir, file), ann_path))
    return [{**tokenizer.encode_plus(text, return_offsets_mapping=True), 'labels': get_labels(tokens, entities)}
            for text, entities in data if (tokens := tokenizer.encode_plus(text, return_offsets_mapping=True)) and len(tokens['input_ids']) <= 2048]

def get_labels(tokens, entities):
    labels = [0] * len(tokens['offset_mapping'])
    for start, end, entity in entities:
        for i, (token_start, token_end) in enumerate(tokens['offset_mapping']):
            if token_start >= start and token_end <= end:
                labels[i] = task_3_entities.index(f"B-{entity}" if token_start == start else f"I-{entity}")
    return labels

def collate(batch):
    tokens, labels = zip(*batch)
    id = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=tokenizer.pad_token_id)
    mask = (id != tokenizer.pad_token_id).long()
    labels = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=-100)
    return {'input_ids': id, 'attention_mask': mask, 'labels': labels}

train_data_with_labels = load('drive/MyDrive/NEREL/train')
train_loader = DataLoader(Dataset(train_data_with_labels), batch_size=4, shuffle=True, collate_fn=collate)
optimizer = AdamW(model.parameters(), lr=1e-5)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        optimizer.zero_grad()
        outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}')

for module in model.modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        prune.l1_unstructured(module, name='weight', amount=0.1)
        prune.remove(module, 'weight')

model.save_pretrained('model')
tokenizer.save_pretrained('model')