# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

#!wget -nc http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
#!unzip -nq spa-eng.zip
#!ls
#!ls spa-eng
#!head spa-eng/spa.txt

eng2spa = {}
for line in open('spa-eng/spa.txt'):
  line = line.rstrip()
  eng, spa = line.split('\t')
  if eng not in eng2spa:
    eng2spa[eng] = []
    eng2spa[eng].append(spa)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
smoother = SmoothingFunction()

sentence_bleu([['hi']], ['hi'], smoothing_function=smoother.method4)

eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
  spa_list_tokens = []
  for text in spa_list:
    tokens = tokenizer.tokenize(text.lower())
    spa_list_tokens.append(tokens)
    eng2spa_tokens[eng] = spa_list_tokens

from transformers import pipeline
translator = pipeline("translation", model = "Helsinki-NLP/opus-mt-en-es")

translator("I like eggs and ham")

eng_phrases = list(eng2spa.keys())
len(eng_phrases)

eng_phrases_subset = eng_phrases[50000:51000]

translations = translator(eng_phrases_subset)

translations[0]

scores = []
for eng, pred in zip(eng_phrases_subset, translations):
  matches = eng2spa_tokens[eng]

  spa_pred = tokenizer.tokenize(pred['translation_text'].lower())
  score = sentence_bleu(matches, spa_pred, smoothing_function=smoother.method4)
  scores.append(score)

import matplotlib.pyplot as plt
plt.hist(scores, bins=50)
plt.savefig('scores')

import numpy as np
np.mean(scores)

def print_random_translation():
  i = np.random.choice(len(eng_phrases_subset))
  eng = eng_phrases_subset[i]
  print("EN:", eng)

  translation = translations[i]['translation_text']
  print("ES:", translation)

  matches = eng2spa[eng]
  print("Matches:", matches)

print_random_translation()
