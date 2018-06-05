from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
import time

nlp2 = spacy.load(r".\spacy_model_v2")
for text, _ in TEST_DATA:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])