# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

from transformers import pipeline
import textwrap
from pprint import pprint

#!wget -nc https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt

lines = [line.rstrip() for line in open('robert_frost.txt')]
lines = [line for line in lines if len(line)>0]

gen = pipeline("text-generation")

pprint(gen(lines[0], num_return_sequences=3, max_length=20))

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

out = gen(lines[0], max_length=30)
print(wrap(out[0]['generated_text']))

prev = 'Two roads diverged in a yellow wood, the most important' + \
'being the one with the giant red car.'
out = gen(prev + '\n' + lines[2], max_length=60)
print(wrap(out[0]['generated_text']))

prev = 'Two roads diverged in a yellow wood, the most important' + \
'being the one with the giant red car.' + \
'And be one traveler, long I stood and stood in' + \
'an all red and red forest, looking back again at its old road.'
out = gen(prev + '\n' + lines[4], max_length= 90)
print(wrap(out[0]['generated_text']))