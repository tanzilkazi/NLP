import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import time

time_taken = 0

#LR_baseilne
def experiment0(corpus):
    # combine title and body into one column
    corpus["title_body"] = corpus["title"] + " " + corpus["body"]
    corpus = corpus.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    corpus["title_body"] = corpus["title_body"].str.lower()
    corpus["title_body"] = corpus["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer()
    corpus_vec = vec.fit_transform(corpus["title_body"])
    corpus_np = corpus_vec.toarray()
    print("input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np)*100))
    return corpus_vec

def experiment1(corpus):
    # combine title and body into one column
    # corpus["title_body"] = corpus["title"] + " " + corpus["body"]
    # corpus = corpus.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    corpus["title"] = corpus["title"].str.lower()
    corpus["title"] = corpus["title"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer(ngram_range=(2, 2))
    corpus_vec = vec.fit_transform(corpus["title"])

    corpus_np = corpus_vec.toarray()
    print("input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np) * 100))
    return corpus_vec

def train_and_test_LR(X,Y):
    LR = LogisticRegressionCV(Cs=10,penalty="l2",solver="sag",multi_class="multinomial",
                              scoring="accuracy",refit=True,n_jobs=3,max_iter=5)
    accuracy = cross_val_score(LR,X,Y,cv=10,n_jobs=3,scoring="accuracy")
    print("max accuracy: {:0.1f}%".format(max(accuracy)*100))
    print("max mean: {:0.1f}%".format(np.mean(accuracy)*100))
    return LR

def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def elapsed_time(s):
    global time_taken
    print(s,": {:0.3f}s".format(time.clock() - time_taken))
    time_taken = time.clock()


if __name__ == "__main__":
    time_taken = time.clock()

    # import input data
    topic_file = pd.read_csv("topic.csv")
    virality_file = pd.read_csv("virality.csv")

    corpus = topic_file

    corpus_formatted = experiment1(corpus)
    print("Shape of vectorized data:",corpus_formatted.shape)
    elapsed_time("Pre-processing data")

    LR = train_and_test_LR(corpus_formatted,corpus["annotation"])
    elapsed_time("To train and evaluate")

    model = LR.fit(corpus_formatted,corpus["annotation"])
    results = model.predict(corpus_formatted)
    elapsed_time("Prediction complete")

    output = topic_file
    output["results"] = pd.Series(results)
    if "title_body" in output.columns:
        output.drop(["title_body"],axis=1)
    output.to_csv("predictions.csv")


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
