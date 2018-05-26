import csv

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import tree2conlltags

# tree = ne_chunk(pos_tag(word_tokenize("apple is a company")))
# print(tree)
#
# iob_tags = tree2conlltags(tree)
# print(iob_tags)

def conll2string(file):
    text = ""
    count = 0
    with open(INPUT_FILE) as file:
        for line in file:
            if len(line) == 1: # newline
                text = text + " "+ '\n'
            else:
                text = text + " " + line.split(sep=" ")[0]
            count = count + 1
            # if count == 60:
            #     break
    return text

def ner_nltk(text):
    ner_list =[]
    for line in text.splitlines():
        tokens = word_tokenize(line)
        pos = pos_tag(tokens)
        ner = tree2conlltags(ne_chunk(pos))
        for tup in ner:
            ner_list.append(tup)
    return ner_list

if __name__ == "__main__":
    INPUT_FILE = r".\conll03\eng.testb"
    text = conll2string(INPUT_FILE)

    print(text)
    print("\n#########\n")
    ners = ner_nltk(text)

    for item in ners:
        print(item[0],item[2])

    print(text)
    print("\n#########\n")

    count = 0
    with open(INPUT_FILE) as file:
        for line in file:
            if line != '\n':
                line = line.strip("\n")
                out = line.split(sep=" ")
                out.append(ners[count][2])
                if out[0] != ners[count][0]:
                    print("\tinput and ner out of sync, exiting\n")
                    print("\tout:",out,"\n\tner:",ners[count])
                    #exit(0)
                count = count + 1
                print(out)