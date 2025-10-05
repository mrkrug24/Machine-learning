import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Iterable, Set, Tuple

LABELS = [
    'AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 
    'EVENT', 'FACILITY', 'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 
    'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 'ORGANIZATION', 'PENALTY', 
    'PERCENT', 'PERSON', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', 
    'TIME', 'WORK_OF_ART'
]

task_3_entities = ['O'] + [f"{p}-{t}" for t in LABELS for p in ['B', 'I']]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx].get('input_ids', []), self.data[idx].get('labels', [])

def get_labels(tokens, entities):
    labels, tracked_type, tracked_start = [], None, None
    for i, (token_position, token_type) in enumerate(zip(tokens['offset_mapping'], entities)):
        token_start, token_end = token_position
        new_type = None if token_type == 'O' else token_type[2:]
        if tracked_type and tracked_type != new_type:
            labels.append((tracked_start, tokens['offset_mapping'][i - 1][1], tracked_type))
        if new_type and new_type != tracked_type:
            tracked_start = token_start
        tracked_type = new_type
    if tracked_type:
        labels.append((tracked_start, token_end, tracked_type))
    return labels

class Solution:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("model")
        self.model = AutoModelForTokenClassification.from_pretrained("model").eval()

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        test_data_tokens = [self.tokenizer.encode_plus(text, return_offsets_mapping=True) for text in texts]
        test_dataset = Dataset(test_data_tokens)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=self.collate)

        with torch.no_grad():
            test_predictions = [self.model(batch['input_ids'], attention_mask=batch['attention_mask']).logits.argmax(dim=-1) for batch in test_loader]

        return [set(get_labels(tokens, [task_3_entities[p] for p in pred[0] if p != -100]))
            for tokens, pred in zip(test_data_tokens, test_predictions)]
        
    def collate(self, batch):
        tokens, labels = zip(*batch)
        id = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask = (id != self.tokenizer.pad_token_id).long()
        labels = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=-100)
        return {'input_ids': id, 'attention_mask': mask, 'labels': labels}