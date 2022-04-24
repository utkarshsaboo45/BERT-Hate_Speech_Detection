import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.preprocessing import OneHotEncoder
from transformers import DistilBertModel, DistilBertTokenizer


DISTIL_MODEL_NAME = "distilbert-base-uncased"  # "distilbert-base-uncased-finetuned-sst-2-english"
DISTIL_TOKENIZER = DistilBertTokenizer.from_pretrained(DISTIL_MODEL_NAME, truncation=True, do_lower_case=True)
DISTIL_TRAIN_BATCH_SIZE = 16
DISTIL_DEV_BATCH_SIZE = 1
DISTIL_TEST_BATCH_SIZE = 1

def get_distil_hyperparams():
    return {
        "MAX_LEN": 128,
        "LEARNING_RATE": 1e-05,
        "TOKENIZER": DISTIL_TOKENIZER,
        "DEVICE": torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
        "TRAIN_PARAMS": {
            "batch_size": DISTIL_TRAIN_BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0
        },
        "DEV_PARAMS": {
            "batch_size": DISTIL_DEV_BATCH_SIZE,
            "shuffle": False,
            "num_workers": 0
        },
        "TEST_PARAMS": {
            "batch_size": DISTIL_TEST_BATCH_SIZE,
            "shuffle": False,
            "num_workers": 0
        },
        "MODEL_PATH": "../models/pytorch_distilbert.bin",
        "VOCAB_PATH": "../models/vocab_distilbert.bin"
    }

class HateDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = OneHotEncoder(sparse=False).fit_transform(np.array(self.data["hatespeech"]).reshape(-1, 1))
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float)
        }

class DistilBERTMultiClass(nn.Module):
    def __init__(self, n_classes, model_name=DISTIL_MODEL_NAME):
        super(DistilBERTMultiClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

