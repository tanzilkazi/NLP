import nltk as nl
import string as str
import random
import os

# file has characters with /n at the end of each line
temp_file = "temp.tex"

# remove lines starting with / and #
with open("w1_astroph-abstracts-sample.tex","r") as input:
    with open(temp_file,"w") as output:
        for line in input:
            if not ('\\' in line[0] or '%' in line[0]):
                output.write(line)

sanitized_f = open(temp_file,"r").read()

# remove punctuation
translator = sanitized_f.maketrans("","",str.punctuation)
file_nopunc = sanitized_f.translate(translator)


#sanitized_f = "the quick brown fox jumps over the lazy dog"

# tokenize
words = nl.word_tokenize(file_nopunc)
# generate bigrams
bigrams = nl.bigrams(words)

# generate conditional freq dist
cfd = nl.ConditionalFreqDist(bigrams)
cpd = nl.ConditionalProbDist(cfd,nl.SimpleGoodTuringProbDist)

# generate text
print_array = []
num = 10

first_w = random.choice(words)
print_array.append(first_w)

# Simple Markov model
# lookup_w = first_w
# for i in range(num):
#     print_array.append(cfd[lookup_w].max())
#     lookup_w = cfd[lookup_w].max()

# Markov model with second word selected depending on their probability
lookup_w = first_w
for i in range(num):
    # get dict of second word
    dict = cfd[lookup_w]
    # stores words in second dict, count depends on the freq of word
    word_dist = []
    for k,v in dict.items():
        for j in range(v):
            word_dist.append(k)
    # select word randomly from word_dist - skewed random lookup since words with more freqs appear more
    next_w = random.choice(word_dist)
    print_array.append(next_w)
    lookup_w = next_w

print(print_array)

try:
    os.remove(temp_file)
except OSError:
    pass