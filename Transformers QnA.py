# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

from transformers import pipeline
qa = pipeline("question-answering")

context = "Today I went to the store to purchase a carton of milk"
question = "What did i buy?"

qa(context=context, question=question)