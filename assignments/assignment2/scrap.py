import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
import time
import warnings

warnings.filterwarnings("ignore")
time_taken = 0

#LR_baseilne
def experiment0(data):
    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    data["title_body"] = data["title_body"].str.lower()
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer()
    corpus_vec = vec.fit_transform(data["title_body"])
    corpus_np = corpus_vec.toarray()
    print("input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np)*100))
    return corpus_vec

def experiment1(corpus):

    corpus["title"] = corpus["title"].str.lower()
    corpus["title"] = corpus["title"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer(ngram_range=(2, 2))
    corpus_vec = vec.fit_transform(corpus["title"])

    corpus_np = corpus_vec.toarray()
    print("input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np) * 100))
    return corpus_vec

def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def elapsed_time(s):
    global time_taken
    print(s,": {:0.3f}s".format(time.clock() - time_taken))
    time_taken = time.clock()

if __name__ == "__main__":
    time_taken = time.clock()

    experiment = {
        '1': experiment0,
        '2': experiment1
    }
    chosen_experiment = experiment['1']

    # import input data
    topic_file = pd.read_csv("topic.csv")
    virality_file = pd.read_csv("virality.csv")
    corpus = topic_file.drop(["annotation"],axis=1,inplace=False)
    corpus_annotation = topic_file["annotation"]

    corpus_formatted = chosen_experiment(corpus)
    print("Shape of vectorized data:",corpus_formatted.shape)
    elapsed_time("Pre-processing data")

    accuracy_vec = []
    LR = LogisticRegression(penalty="l2", solver="sag", multi_class="multinomial",
                            warm_start=True, n_jobs=-1, max_iter=20)
    cv = ShuffleSplit(n_splits=10, train_size=0.9, test_size=None)
    num_loops = 0
    while (num_loops < 5):
        accuracy = cross_val_score(LR, corpus_formatted, corpus_annotation, cv=cv, n_jobs=-1, scoring="accuracy")
        accuracy_vec.append(np.mean(accuracy) * 100)
        num_loops = num_loops + 1

    print("Mean accuracy: {:0.1f} +- {:0.1f}".format(np.mean(accuracy_vec),np.std(accuracy_vec)))
    elapsed_time("To train and evaluate")

    train_X, test_X, train_Y, test_Y = train_test_split(corpus,corpus_annotation,train_size=0.9,random_state=2)
    corpus_sparse = chosen_experiment(corpus)
    model = LR.fit(corpus_sparse[train_X.index.tolist()],train_Y)
    results = model.predict(corpus_sparse[test_X.index.tolist()])
    elapsed_time("Prediction complete")

    output_df = pd.DataFrame(corpus.iloc[test_X.index.tolist(),])
    output_df["annotation"] = corpus_annotation[test_X.index.tolist()]
    output_df["results"] = pd.Series(results,index=test_X.index.tolist())
    if "title_body" in output_df.columns:
        output_df.drop(["title_body"],axis=1)
    output_df.to_csv("predictions.csv")


    # for train_index,test_index in Kf.split(corpus_np):
    #     corpus_scaled = preprocessing.scale(corpus_np[train_index])
    #     model = LR.fit(corpus_scaled,corpus["annotation"][train_index])
    #     print("Time for fold {:0.0f}: {:0.3f}s".format(i,time.clock() - time_taken))
    #     time_taken = time.clock()
    #     i = i + 1



