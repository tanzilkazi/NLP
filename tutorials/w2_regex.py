import nltk as nl
import re

wordlist = [w for w in nl.corpus.words.words('en')]

# alpha = [w for w in wordlist if re.search('^[a-z]+$',w)]
# print("all alpha:", alpha)
#
# atod = [w for w in wordlist if re.search('^[a-d]+$',w)]
# print("only letters a,b,c and d:", atod)
#
# endb = [w for w in wordlist if re.search('^[a-z]+b$',w)]
# print("lowercase, end with b:", endb)
#
# bab = [w for w in wordlist if re.search(r"bab",w)]
# print("string with bab:", bab)

repeat_letters = [w for w in wordlist if re.search(r"([a-z])\1",w)]
print("repeated letters:",repeat_letters[:10])

jo = [re.sub(r"jo","AA",w) for w in wordlist if re.search(r"jo",w)]
print(jo[:10])

