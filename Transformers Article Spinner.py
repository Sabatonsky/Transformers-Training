# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

from transformers import pipeline
import pandas as pd
import numpy as np
import textwrap

#!wget -nc https://www.dropbox.com/s/7hb8bwbtjmxovlc/bbc_text_cls.csv

df = pd.read_csv('bbc_text_cls.csv')

df.info()

labels = set(df['labels'])
print(labels)

label = 'business'
texts = df.loc[df.labels == label,:]['text']

i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]

print(textwrap.fill(doc, replace_whitespace = False, fix_sentence_endings = True))

mlm = pipeline('fill-mask')

mlm('BMW <mask> new model pipeline')
