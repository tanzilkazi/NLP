import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string
from sklearn.model_selection import KFold

if __name__ == "__main__":
    # import input data
    topic_file = pd.read_csv("topic.csv")
    virality_file = pd.read_csv("virality.csv")

    # combine title and body into one column
    corpus = topic_file
    corpus["title_body"] = corpus["title"] + " " + corpus["body"]
    corpus = corpus.drop(["title", "body"], axis=1)


    # lower case and remove punctuations
    corpus["title_body"] = corpus["title_body"].str.lower()
    #corpus["title_body"] = corpus["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer()
    corpus_vec = vec.fit_transform(corpus["title_body"])
    print(corpus_vec.toarray().shape)




    # #tester
    # d = {'id':[1,2,3],
    #      'str':["today is sunday","nice! weather is is great","sunday funday"]}
    #
    # df = pd.DataFrame(data=d)
    # vec = CountVectorizer()
    # print(df)
    # x = vec.fit_transform(df["str"])
    # print(x.toarray())