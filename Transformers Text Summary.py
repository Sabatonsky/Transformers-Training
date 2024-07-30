# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

#!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv

import pandas as pd
import textwrap
from transformers import pipeline

df = pd.read_csv('bbc_text_cls.csv')

doc = df[df.labels == 'business']['text'].sample(random_state=42)

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

print(wrap(doc.iloc[0]))

summarizer = pipeline("summarization")

summarizer(doc.iloc[0].split("\n", 1)[1])

def print_summary(doc):
  result = summarizer(doc.iloc[0].split("\n", 1)[1])
  print(wrap(result[0]['summary_text']))

doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
print_summary(doc)

print(wrap(doc.iloc[0]))
