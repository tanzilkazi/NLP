import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
import sklearn.metrics as mets
import time
import warnings
from nltk.corpus import stopwords
import nltk as nl
from nltk.tokenize import WhitespaceTokenizer

warnings.filterwarnings("ignore")
time_taken = 0

#LR_baseilne
def experiment0(data):
    print("Running experiment 0...")
    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    data["title_body"] = data["title_body"].str.lower()
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec = CountVectorizer()
    corpus_vec = vec.fit_transform(data["title_body"])
    corpus_np = corpus_vec.toarray()
    print("Input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np)*100))
    return corpus_vec

def experiment1(data):
    print("Running experiment 1...")
    data["title"] = data["title"].str.lower()
    data["title"] = data["title"].str.replace('[{}]'.format(string.punctuation), '')

    sent_list = data["body"].str.split("\n").values
    data["first_sent"] = pd.Series([x[0] for x in sent_list])
    data["first_sent"] = data["first_sent"].str.lower()
    data["first_sent"] = data["first_sent"].str.replace('[{}]'.format(string.punctuation), '')
    vec = CountVectorizer(ngram_range=(3, 3))
    corpus_vec = vec.fit_transform(data["title"] + " " + data["first_sent"])

    corpus_np = corpus_vec.toarray()
    print("Input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np) * 100))

    return corpus_vec

def experiment2(data):
    print("Running experiment 2...")
    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec=TfidfVectorizer(lowercase=True,stop_words="english")
    corpus_vec = vec.fit_transform(data["title_body"])
    return corpus_vec

def experiment3(data):
    print("Running experiment 3...")
    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    tokenized = data["title_body"].apply(WhitespaceTokenizer().tokenize)
    no_stops = tokenized.apply(lambda title_body: [word for word in title_body if word not in stopwords.words("english")])
    # remove punctuations
    translator = str.maketrans('', '', string.punctuation)

    data["title_body"] = no_stops.apply(lambda documents: [word.translate(translator) for word in documents])
    data["title_body"] = data["title_body"].apply(lambda listt: " ".join(listt))
    vec=TfidfVectorizer(lowercase=True)
    corpus_vec = vec.fit_transform(data["title_body"])
    return corpus_vec

def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def elapsed_time(s):
    global time_taken
    print(s,": {:0.1f}s".format(time.clock() - time_taken))
    time_taken = time.clock()

if __name__ == "__main__":
    time_taken = time.clock()
    start_time = time_taken

    experiment = {
        "0": experiment0,
        "1": experiment1,
        "2": experiment2,
        "3": experiment3
    }
    chosen_experiment = experiment["3"]

    # import input data
    topic_file = pd.read_csv("topic.csv")
    virality_file = pd.read_csv("virality.csv")
    corpus = topic_file.drop(["annotation"],axis=1,inplace=False)
    corpus_annotation = topic_file["annotation"]

    corpus_formatted = chosen_experiment(corpus)
    print("Shape of vectorized data:", corpus_formatted.shape)
    elapsed_time("Pre-processing data")

    accuracy_vec = []
    LR = LogisticRegression(penalty="l2", solver="sag", multi_class="multinomial",
                            warm_start=True, n_jobs=-1, max_iter=30)
    cv = ShuffleSplit(n_splits=10, train_size=0.9)
    num_loops = 0
    while (num_loops < 10):
        accuracy = cross_val_score(LR, corpus_formatted, corpus_annotation, cv=cv, n_jobs=-1, scoring="accuracy")
        accuracy_vec.append(np.mean(accuracy) * 100)
        num_loops = num_loops + 1

    print("Mean accuracy: {:0.1f} +- {:0.1f}".format(np.mean(accuracy_vec),np.std(accuracy_vec)))
    elapsed_time("To train and evaluate")

    train_X, test_X, train_Y, test_Y = train_test_split(corpus,corpus_annotation,train_size=0.9,random_state=2)
    corpus_sparse = chosen_experiment(corpus)
    model = LR.fit(corpus_sparse[train_X.index.tolist()],train_Y)
    results = model.predict(corpus_sparse[test_X.index.tolist()])
    print("Prediction accuracy: {:0.1f}".format(mets.accuracy_score(test_Y,results)*100))
    elapsed_time("Prediction complete")

    output_df = pd.DataFrame(corpus.iloc[test_X.index.tolist(),])
    output_df["annotation"] = corpus_annotation[test_X.index.tolist()]
    output_df["results"] = pd.Series(results,index=test_X.index.tolist())
    if "title_body" in output_df.columns:
        output_df.drop(["title_body"],axis=1)
    output_df.to_csv("predictions.csv")

    print("Total time taken: {:0.1f}s".format(time.clock() - start_time))

    # for train_index,test_index in Kf.split(corpus_np):
    #     corpus_scaled = preprocessing.scale(corpus_np[train_index])
    #     model = LR.fit(corpus_scaled,corpus["annotation"][train_index])
    #     print("Time for fold {:0.0f}: {:0.3f}s".format(i,time.clock() - time_taken))
    #     time_taken = time.clock()
    #     i = i + 1



