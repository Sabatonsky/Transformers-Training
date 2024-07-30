# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

#!pip install transformers datasets

from datasets import load_dataset

raw_datasets = load_dataset("glue", "sst2")

raw_datasets

raw_datasets = load_dataset('glue', 'sst2')

raw_datasets['train'].features

from transformers import AutoTokenizer

checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_sentences = tokenizer(raw_datasets['train'][0:3]['sentence'])
from pprint import pprint
pprint(tokenized_sentences)

def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

from transformers import TrainingArguments

#!pip install accelerate -U

training_args = TrainingArguments(
    'my_trainer',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=1
)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2
)
