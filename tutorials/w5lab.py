
#from __future__ import print_function, division
from collections import defaultdict, Counter
import os
import random
import math
from nltk.corpus import movie_reviews

# ROOT = os.path.expanduser('~/nltk_data/corpora/movie_reviews')
# collect (class, filename) pairs
pos = [doc_id for doc_id in movie_reviews.fileids(categories="pos")]
neg = [doc_id for doc_id in movie_reviews.fileids(categories="neg")]

# neg = [('neg', ROOT + '/neg/' + f)
#        for f in os.listdir(ROOT + '/neg')]
files = pos + neg
print(files)
#
# hold out 20% of the data
random.shuffle(files)
test = files[:len(files) // 5]
train = files[len(files) // 5:]
# test

feature_counts = defaultdict(float)
prior = defaultdict(float)
ndocs = 0

for klass, filename in train:
    # get the bag of words in each file
    tokens = open(filename, 'rU').read().lower().split()
    text_counts = Counter(tokens)

    # TODO #3: handle arbitrary features

    for feat in text_counts:
        # record how often each word co-occurs with a particular class label
        feature_counts[feat, klass] += text_counts[feat]
    prior[klass] += 1
    ndocs += 1


ncorrect = 0
nexamples = 0
for correct, filename in test:
    text_counts = Counter(open(filename, 'rU').read().lower().split())
    log_proba = defaultdict(float)
    for klass in prior.keys():
        # add log probability of class (i.e. prior probability):
        log_proba[klass] += math.log(prior[klass] / ndocs)

        # add log probability of document given class:

        # TODO #1: add log probability for each found feature given class
        # TODO #2: implement Laplace (or Lidstone) smoothing by assuming
        #          a fake document in each class including all features
        #          and another with no features. Note this should not
        #          affect the prior.
        #for feat in text_counts:
        # TODO: log_proba[klass] += ...

    # order the classes by descending probability
    log_proba = [(v, k) for k, v in log_proba.items()]
    log_proba.sort(reverse=True)
    print(correct, log_proba[0])
    if log_proba[0][1] == correct:
        ncorrect += 1
    nexamples += 1

print(ncorrect, "of", nexamples, "correct")