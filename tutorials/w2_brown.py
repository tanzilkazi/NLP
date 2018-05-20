import nltk as nl
import re

brown = nl.corpus.brown.words()
hyphen = [(p, w, n) for p, w, n in nl.ngrams(brown, 3) if len(w) > 2 and '-' in w]
print(hyphen)
for w in hyphen:
    if re.search(r'-',w):
        re.sub(r'-'," ",w)
print(hyphen)

