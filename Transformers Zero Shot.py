# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

from transformers import pipeline

import numpy as np
import pandas as pd
import textwrap

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

classifier = pipeline('zero-shot-classification')

classifier('This is a great movie', candidate_labels=['positive', 'negative'])

#!wget -nc https://www.dropbox.com/s/7hb8bwbtjmxovlc/bbc_text_cls.csv

df = pd.read_csv('bbc_text_cls.csv')

labels = list(set(df['labels']))

print(textwrap.fill(df.iloc[1024]['text']))

print(df.iloc[1024]['labels'])

preds = classifier(df.iloc[500:600]['text'].tolist(), candidate_labels=labels)

predicted_labels = [d['labels'][0] for d in preds]

df['predicted_labels'] = predicted_labels

print("acc", np.mean(df['predicted_labels'] == df['labels']))

N = len(df)
K = len(labels)
label2idx = {v:k for k,v in enumerate(labels)}

probs = np.zeros((N, K))
for i in range(N):
  d = preds[i]
  for label, score in zip(d['labels'], d['scores']):
    k = label2idx[label]
    probs[i, k] = score

int_labels = [label2idx[x] for x in df['labels']]
int_preds = np.argmax(probs, axis=1)
cm = confusion_matrix(int_labels, int_preds, normalize='true')
print(cm)

f1_score = f1_score(df['labels'], predicted_labels, average='micro')
roc_auc = roc_auc_score(int_labels, probs, multi_class='ovo')
print(f1_score)
print(roc_auc)
