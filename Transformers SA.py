# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

from transformers import pipeline

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

current_device = torch.cuda.current_device()
classifier = pipeline("sentiment-analysis", device = current_device)

#!wget -nc https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/twitter_us_airline_sentiment_analysis.csv

df_ = pd.read_csv("twitter_us_airline_sentiment_analysis.csv")
df = df_.loc[:,['airline_sentiment', 'text']].copy()
df = df[df.airline_sentiment != 'neutral'].copy()

target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)

texts = df['text'].tolist()
predictions = classifier(texts)

print(predictions[:5])

probs = [d['score'] if d['label'].startswith('P') else 1-d['score'] for d in predictions]

preds = [1 if d['label'].startswith('P') else 0 for d in predictions]
preds = np.array(preds)

print("acc:", np.mean(df.target == preds))

cm = confusion_matrix(df['target'], preds, normalize='true')
print(cm)

tick_labels = ['negative', 'positive']
sn.heatmap(cm, annot = True, fmt = 'g', xticklabels=tick_labels, yticklabels=tick_labels)
plt.savefig('confusion_matrix')

f1_pos = f1_score(df['target'], preds)
f1_neg = f1_score(1-df['target'], 1-preds)
print(f1_pos, f1_neg)

roc_auc = roc_auc_score(df['target'], probs)
print(roc_auc)
