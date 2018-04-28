#!/usr/bin/env python

# Implementation of Multinomial Naive Bayes.
# see also:
# https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

from __future__ import print_function, division
from collections import defaultdict, Counter
import os
import random
import math

# ROOT = os.path.expanduser('~/nltk_data/corpora/movie_reviews')
# # collect (class, filename) pairs
# pos = [('pos', ROOT + '/pos/' + f)
#        for f in os.listdir(ROOT + '/pos')]
# neg = [('neg', ROOT + '/neg/' + f)
#        for f in os.listdir(ROOT + '/neg')]
# files = pos + neg
#
ROOT = os.path.expanduser(r"C:\Users\kazit\AppData\Roaming\nltk_data\corpora\movie_reviews")
# collect (class, filename) pairs
pos = [('pos', ROOT + "\\pos\\" + f)
       for f in os.listdir(ROOT + '\\pos')]
neg = [('neg', ROOT + '\\neg\\' + f)
       for f in os.listdir(ROOT + '\\neg')]
files = pos + neg

files = pos + neg

# hold out 20% of the data
random.shuffle(files)
test = files[:len(files) // 5]
train = files[len(files) // 5:]

feature_names = set()
feature_counts = Counter()
klass_totals = Counter()
prior = Counter()
ndocs = 0


def extract_features(filename):
    return Counter(open(filename, 'rU').read().lower().split())


for klass, filename in train:
    # get the bag of words in each file
    text_counts = extract_features(filename)
    # TODO #3: handle arbitrary features
    for feat in text_counts:
        # record how often each word co-occurs with a particular class label
        feature_counts[feat, klass] += text_counts[feat]
        klass_totals[klass] += text_counts[feat]
        feature_names.add(feat)
    prior[klass] += 1
    ndocs += 1

smoothing_const = 1
n_features = len(feature_names)

ncorrect = 0
nexamples = 0
for correct, filename in test:
    text_counts = extract_features(filename)
    log_proba = defaultdict(float)
    for klass in prior.keys():
        # add log probability component from prior:
        log_proba[klass] += math.log(prior[klass] / ndocs)
        # add log probability component per feature
        for feat in text_counts:
            p_i_klass = ((feature_counts[feat, klass] + smoothing_const) /
                         (klass_totals[klass] + smoothing_const * n_features))
            log_proba[klass] += text_counts[feat] * math.log(p_i_klass)

    # order the classes by descending probability
    log_proba = [(v, k) for k, v in log_proba.items()]
    log_proba.sort(reverse=True)
    print(correct, log_proba[0])
    if log_proba[0][1] == correct:
        ncorrect += 1
    nexamples += 1

print(ncorrect, "of", nexamples, "correct")