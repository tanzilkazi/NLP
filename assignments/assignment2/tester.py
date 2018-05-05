import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import defaultdict
import os
#tester
# d = {'id':[1,2,3,4,5,6,7,8,9,10],
#      'str':["today is sunday",
#             "nice! weather is is great",
#             "sunday funday",
#             "what's happening tomorrow?",
#             "got a meeting on Monday",
#             "when's the next public holiday?",
#             "Wednesday",
#             "let us meet tomorrow",
#             "it will rain tomorrow",
#             "tomorrow is Monday"],
#      "cat":["A","A","A","B","A","B","B","B","B","A"]}
#
# df = pd.DataFrame(data=d)
# vec=TfidfVectorizer(lowercase=True,stop_words="english")
# fit = vec.fit_transform(df["str"])
#

import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(u'The President of the United States')

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)