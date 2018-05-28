from nltk import word_tokenize, pos_tag, ne_chunk, tokenize
from nltk import tree2conlltags
from _collections import defaultdict

BREAK_COUNT = 20

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
            if count == BREAK_COUNT:
                break
    return text

def ner_nltk(text):
    ner_list =[]
    for line in text.splitlines():
        tokens = line.strip().split(sep=" ")
        pos = pos_tag(tokens)
        ner = tree2conlltags(ne_chunk(pos))
        for tup in ner:
            ner_list.append(tup)
    return ner_list

def write2file(data,file):
    with open(file,"w") as f:
        for tag in data:
            str_line = ""
            for item in tag:
                str_line = str_line + " " + str(item)
            str_line = str_line.strip()
            if str_line != "\n":
                str_line = str_line + "\n"
            f.write(str_line)
    print("Output written to file "+ file)
    return True

def output_format(in_file,named_ents):
    count_ents = 0
    output = []
    break_count = 0
    with open(in_file) as file:
        for line in file:
            if line == '\n':
                output.append('\n')
            else:
                line = line.strip("\n")
                out = line.split(sep=" ")
                out.append(named_ents[count_ents][-1])
                if out[0] != named_ents[count_ents][0]:
                    print("\tinput and ner out of sync, exiting\n")
                    print("\tout:", out, "\n\tner:", named_ents[count_ents])
                    count_ents = count_ents + 1
                output.append(out)
            break_count = break_count + 1
            if break_count == BREAK_COUNT:
                break
    #output = nltk2conll_mapper(output)
    return output


def nltk2conll_mapper(data):
    mapper = {"B-FACILITY": "B-MISC",
              "B-GPE": "B-LOC",
              "B-GSP": "B-LOC",
              "B-LOCATION": "B-LOC",
              "B-ORGANIZATION": "B-ORG",
              "B-PERSON": "B-PER",
              "I-FACILITY": "I-MISC",
              "I-GPE": "I-LOC",
              "I-GSP": "I-LOC",
              "I-LOCATION": "I-LOC",
              "I-ORGANIZATION": "I-ORG",
              "I-PERSON": "I-PER",
              "O": "O"
              }
    index = 0
    while index < len(data):
        if(data[index][-1] in mapper):
            data[index][-1] = mapper[data[index][-1]]
        index = index + 1
    return data

if __name__ == "__main__":
    INPUT_FILE = r".\conll03\eng.testb"
    OUTPUT_FILE = "nltk_output1.txt"
    text = conll2string(INPUT_FILE)
    ners = ner_nltk(text)
    out = output_format(INPUT_FILE,ners)
    print(out)
    #write2file(out,OUTPUT_FILE)