import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import sklearn.metrics as mets
import time
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
import spacy as sp
from scipy.sparse import  hstack
import seaborn as sb
import matplotlib.pyplot as plot

warnings.filterwarnings("ignore")
time_taken = 0

# experiment0: bag of words
def experiment0(data):
    print("Running experiment 0...")

    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    data["title_body"] = data["title_body"].str.lower()
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    # vectorize
    vec = CountVectorizer()
    corpus_vec = vec.fit_transform(data["title_body"])
    corpus_np = corpus_vec.toarray()
    print("Input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np)*100))
    return corpus_vec

# experiment1: uni + bigrams for title and leading paragragh
def experiment1(data):
    print("Running experiment 1...")

    data["title"] = data["title"].str.lower()
    data["title"] = data["title"].str.replace('[{}]'.format(string.punctuation), '')

    # extract first paragraph
    sent_list = data["body"].str.split("\n").values
    data["first_para"] = pd.Series([x[0] for x in sent_list])
    data["first_para"] = data["first_para"].str.lower()
    data["first_para"] = data["first_para"].str.replace('[{}]'.format(string.punctuation), '')

    # vectorize with uni- and bi- grams
    vec = CountVectorizer(ngram_range=(1, 2),max_features=12000)
    corpus_vec = vec.fit_transform(data["title"] + " " + data["first_para"])

    corpus_np = corpus_vec.toarray()
    print("Input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_np) * 100))

    return corpus_vec

# experiment2: stopwords + tfidf over entire title and body of text
def experiment2(data):
    print("Running experiment 2...")

    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    # lower case and remove punctuations
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    # vectorize as tf-idf
    vec=TfidfVectorizer(lowercase=True,stop_words="english", min_df = 0.01,max_df=0.5)
    corpus_vec = vec.fit_transform(data["title_body"])

    return corpus_vec

# experiment3: stopwords + tfidf over entire title and body of text in the following order: tokenize with whitespace,
# remove stopwords, remove punctuation, vectorize
def experiment3(data):
    print("Running experiment 3...")
    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)

    # tokeinze
    tokenized = data["title_body"].apply(WhitespaceTokenizer().tokenize)
    # remove stop words
    no_stops = tokenized.apply(lambda title_body: [word for word in title_body if word not in stopwords.words("english")])
    # remove punctuations
    translator = str.maketrans('', '', string.punctuation)

    # vectorize with tf-idf
    data["title_body"] = no_stops.apply(lambda documents: [word.translate(translator) for word in documents])
    data["title_body"] = data["title_body"].apply(lambda listt: " ".join(listt))
    vec=TfidfVectorizer(lowercase=True,min_df = 0.01,max_df=0.5)
    corpus_vec = vec.fit_transform(data["title_body"])
    return corpus_vec

# experiment4: tfidf, stopwords and uni- and bi-grams with 12000 features
def experiment4(data):
    print("Running experiment 4...")

    # combine title and body into one column
    data["title_body"] = data["title"] + " " + data["body"]
    data = data.drop(["title", "body"], axis=1)
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')

    vec = TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1, 2),max_features=12000)
    corpus_vec = vec.fit_transform(data["title_body"])
    return corpus_vec

# experiment5: stopwords, entity detection and tf-idf with uni- and bi-grams
def experiment5(data):
    print("Running experiment 5...")
    nlp = sp.load('en_core_web_sm')

    data["title_body"] = data["title"] + " " + data["body"]
    data["title_body"] = data["title_body"].str.replace('[{}]'.format(string.punctuation), '')
    data = data.drop(["title", "body"], axis=1)

    entity_list = []
    for index, row in data["title_body"].iteritems():
        str = ""
        doc = nlp(row)
        for ent in doc.ents:
            str += "_"+ ent.label_+"_ "
        entity_list.append(str)

    data["ents"] = pd.Series(entity_list)

    vec_ents = TfidfVectorizer(lowercase=True, stop_words="english")
    vec_bags = TfidfVectorizer(lowercase=True, stop_words="english",max_features=1000)
    corpus_ents = vec_ents.fit_transform(data["ents"])
    corpus_bags = vec_bags.fit_transform(data["title_body"])
    corpus_vec = hstack([corpus_bags,corpus_ents]).toarray()
    print("Input sparsity ratio: {:0.1f}%".format(sparsity_ratio(corpus_vec ) * 100))
    return corpus_vec

# calculate sparsity ratio for sparse matrices
def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

# elapsed time between now and previous call
def elapsed_time(s):
    global time_taken
    print(s,": {:0.1f}s".format(time.clock() - time_taken))
    time_taken = time.clock()

if __name__ == "__main__":
    time_taken = time.clock()
    start_time = time_taken

    # select which experiment to run
    exp_to_run = "2"

    experiment = {
        "0": experiment0,
        "1": experiment1,
        "2": experiment2,
        "3": experiment3,
        "4": experiment4,
        "5": experiment5
    }
    chosen_experiment = experiment[exp_to_run]

    # import input data
    topic_file = pd.read_csv("topic.csv")
    virality_file = pd.read_csv("virality.csv")

    corpus = topic_file.drop(["annotation"],axis=1,inplace=False)
    corpus_annotation = topic_file["annotation"]

    # corpus = virality_file.drop(["class"], axis=1, inplace=False)
    # corpus_annotation = virality_file["class"]

    corpus_formatted = chosen_experiment(corpus)
    print("Shape of vectorized data:", corpus_formatted.shape)
    elapsed_time("Time to vectorize data")

    accuracy_vec = []
    f1_weighted_vec = []
    LR = LogisticRegression(penalty="l2", solver="sag", multi_class="multinomial",
                            warm_start=True, n_jobs=-1, max_iter=20)
    cv = ShuffleSplit(n_splits=10, train_size=0.9)
    num_loops = 0

    # perform 10-fold cross validation 10 times and calculate the average accuracy andd weighted f1-score
    while (num_loops < 10):
        # CROSS VALIDATION
        accuracy = cross_validate(LR, corpus_formatted, corpus_annotation, cv=cv, n_jobs=-1, scoring=["accuracy","f1_weighted"])
        accuracy_vec.append(np.mean(accuracy['test_accuracy']) * 100)
        f1_weighted_vec.append(np.mean(accuracy['test_f1_weighted']) * 100)
        num_loops = num_loops + 1

    print("Mean accuracy: {:0.1f} +- {:0.1f}".format(np.mean(accuracy_vec),np.std(accuracy_vec)))
    print("Mean f1-weighted: {:0.1f} +- {:0.1f}".format(np.mean(f1_weighted_vec), np.std(f1_weighted_vec)))

    elapsed_time("Time to train and evaluate")

    print("\nModel validation:")

    topic_classes = ["Business", "Entertainment", "Error", "Health", "Other", "Politics", "Science and Technology", "Society", "Sports", "War"]
    virality_classes = ["high", "low"]

    # use 90% of the data to train and test on remaining 10% of data
    train_X, test_X, train_Y, test_Y = train_test_split(corpus,corpus_annotation,train_size=0.9,random_state=2)
    corpus_sparse = chosen_experiment(corpus)
    model = LR.fit(corpus_sparse[train_X.index.tolist()],train_Y)
    results = model.predict(corpus_sparse[test_X.index.tolist()])

    print("Overall prediction accuracy: {:0.1f}".format(mets.accuracy_score(test_Y,results)*100))
    print("Overall prediction f1-weighted: {:0.1f}".format(mets.f1_score(test_Y, results, average="weighted") * 100))
    print(mets.classification_report(test_Y, results, target_names=topic_classes))

    elapsed_time("Time to complete prediction")

    # print to file
    output_df = pd.DataFrame(corpus.iloc[test_X.index.tolist(),])
    output_df["annotation"] = corpus_annotation[test_X.index.tolist()]
    output_df["results"] = pd.Series(results,index=test_X.index.tolist())
    if "title_body" in output_df.columns:
        output_df.drop(["title_body"],axis=1)
    output_df.to_csv("predictions"+exp_to_run+".csv")

    # Confusion Matrix
    conf_matrix = mets.confusion_matrix(test_Y, results, labels=topic_classes)
    df_conf_matrix = pd.DataFrame(conf_matrix, index=topic_classes, columns=topic_classes)
    fig = plot.figure(figsize=(10, 10),tight_layout=True)
    conf_map = sb.heatmap(df_conf_matrix, annot=True, cmap="Blues", linewidths=0.5, linecolor="black", cbar=False,
                          square=True)
    plot.xticks(rotation=90)
    plot.yticks(rotation=0)
    plot.xlabel("Predicted classes")
    plot.ylabel("True classes")

    print("Total time taken: {:0.1f}s".format(time.clock() - start_time))
    plot.savefig("predictions_conf"+exp_to_run+".png")
    print("\n")




