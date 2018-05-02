import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

#tester
d = {'id':[1,2,3,4,5,6,7,8,9,10],
     'str':["today is sunday",
            "nice! weather is is great",
            "sunday funday",
            "what's happening tomorrow?",
            "got a meeting on Monday",
            "when's the next public holiday?",
            "Wednesday",
            "let us meet tomorrow",
            "it will rain tomorrow",
            "tomorrow is Monday"],
     "cat":["A","A","A","B","A","B","B","B","B","A"]}

df = pd.DataFrame(data=d)
# vec = CountVectorizer()
# x = vec.fit_transform(df["str"])
# Kf = KFold(n_splits=3,shuffle=True)
# for train_index,test_index in Kf.split(df):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print(df["cat"][test_index])
#

topic_file = pd.read_csv("topic.csv")
body = topic_file["body"]
print(topic_file["body"][41])
print(repr(topic_file["body"][41]))