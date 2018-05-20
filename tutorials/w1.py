import string as str
from collections import defaultdict
import collections as col
import math as math

# ### Remove punctuations
# s = "What %^&*are you doing? $%^&*"
#
### s becomes an array of chars
# translator = s.maketrans("","",str.punctuation)
#
# print("1. ",s.translate(translator))
#
# temp = ""
# for char in s:
#     if char not in str.punctuation:
#         temp = temp + char
# print("2. ",temp)
#
# ### Split strings - on spaces
# print(s.split())
#
# ## Removing punctuations
# words = [w.strip(str.punctuation) for w in s.split()]
# print(len(words))
#
#
# ###  Collocatations and t-test
# import collections as col
# import math as math
#
# def ttest (nu1,nu2,nb,N):
#     pu1 = nu1/N
#     pu2 = nu2/N
#     pb = nb/N
#     return (pb - (pu1*pu2))/math.sqrt(pb/N)
#
# file = open("w1 pandp12.txt","r").read()
#
# translator = file.maketrans("","",str.punctuation)
# file_nopunc = file.translate(translator)
# words  = file_nopunc.split()
# nwords = len(col.Counter(words))
#
# unigrams = defaultdict(int)
# bigrams = defaultdict(int)
# for w1,w2 in zip(words,words[1:]):
#     unigrams[w1] += 1
#     bigrams[w1,w2] += 1
# unigrams[w2] += 1
#
# scores = []
# for w1,w2 in bigrams:
#     score = ttest(unigrams[w1],unigrams[w2],bigrams[w1,w2],nwords)
#     scores.append((score,w1,w2))
#
# for score, w1, w2 in sorted(scores,reverse=True)[:100]:
#     print(score, w1, w2)

###
import nltk as nl
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

#from nltk.book import text1

file = open("w1 pandp12.txt","r").read()
print("Characters with spaces: ",len(file))
print("Unique item types: ",len(sorted(set(nl.word_tokenize(file))))) # with punctuations
print("Unique words: ", len(set(word.lower() for word in nl.word_tokenize(file) if word.isalpha()))) # without punctuations

s1=file
s2=set(nl.word_tokenize(file)) # all unique words

words = [word.lower() for word in nl.word_tokenize(file) if word.isalpha()]
bigrams = nl.bigrams(words)

big_fd = nl.FreqDist(bigrams)
print ("Bigrams:",big_fd.most_common(20))


### removing stopwords
words_without_stopwords = [word.lower() for word in nl.word_tokenize(file) if word.isalpha() and word not in sw]

bigrams_w_sw = nl.bigrams(words_without_stopwords)

big_fd_w_sw = nl.FreqDist(bigrams_w_sw)
print ("Bigrams, without stop words:",big_fd_w_sw.most_common(20))

