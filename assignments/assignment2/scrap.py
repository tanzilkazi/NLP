import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import time

if __name__ == "__main__":
    time_taken = time.clock()

    # import input data
    topic_file = pd.read_csv("topic.csv")
    virality_file = pd.read_csv("virality.csv")

    # combine title and body into one column
    corpus = topic_file
    corpus["title_body"] = corpus["title"] + " " + corpus["body"]
    corpus = corpus.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    corpus["title_body"] = corpus["title_body"].str.lower()
    corpus["title_body"] = corpus["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer()
    corpus_vec = vec.fit_transform(corpus["title_body"])
    corpus_np = corpus_vec.toarray()
    corpus_scaled = preprocessing.scale(corpus_np)
    print("Preprocessing data: {:0.3f}s".format(time.clock() - time_taken))
    time_taken = time.clock()

    # cross validation
    #Kf = KFold(n_splits=10,shuffle=True)
    LR = LogisticRegression(penalty="l2",
                            solver="sag",
                            multi_class="ovr",
                            warm_start=True,
                            n_jobs=-1,
                            max_iter=3,
                            tol=0.1
                            )
    i = 0
    print(cross_val_score(LR,
                          corpus_scaled,
                          corpus["annotation"],
                          cv=10,
                          n_jobs=-1,
                          scoring="accuracy"))
    print("Predicted: {:0.3f}s".format(time.clock() - time_taken))
    time_taken = time.clock()

    # for train_index,test_index in Kf.split(corpus_np):
    #     corpus_scaled = preprocessing.scale(corpus_np[train_index])
    #     model = LR.fit(corpus_scaled,corpus["annotation"][train_index])
    #     print("Time for fold {:0.0f}: {:0.3f}s".format(i,time.clock() - time_taken))
    #     time_taken = time.clock()
    #     i = i + 1


    # #tester
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
    # vec = CountVectorizer()
    # #print(df)
    # x = vec.fit_transform(df["str"])
    # #print(x.toarray())
    # Kf = KFold(n_splits=3,shuffle=True)
    # for train_index,test_index in Kf.split(df):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     print(df["cat"][test_index])
