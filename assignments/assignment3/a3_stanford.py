import os
import nltk as nl

STANFORD_NER = r".\stanford\stanford-ner-2018-02-27\stanford-ner.jar"
STANFORD_POS = r".\stanford\stanford-postagger-full-2018-02-27\stanford-postagger.jar"
STANFORD_PARSER = r".\stanford\stanford-parser-full-2018-02-27\stanford-parser.jar"
os.environ['JAVAHOME'] = r"C:\Program Files\Java\jre1.8.0_91\bin\java.exe"
os.environ['CLASSPATH'] = STANFORD_NER;STANFORD_POS;STANFORD_PARSER

BREAK_COUNT = 500

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
            # if count == BREAK_COUNT:
            #     break
    return text

def ner_stanford(text):
    ner_list =[]
    st_ner = nl.tag.StanfordNERTagger(r".\stanford\stanford-english-corenlp-2018-02-27-models\edu\stanford\nlp\models\ner\english.conll.4class.distsim.crf.ser.gz")

    for line in text.splitlines():
        tokens = line.strip().split(sep=" ")
        ner = st_ner.tag(tokens)
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
            # if break_count == BREAK_COUNT:
            #     break
    output = nltk2conll_mapper(output)
    return output

def nltk2conll_mapper(data):
    mapper = {"MISC": "I-MISC",
              "ORGANIZATION": "I-ORG",
              "PERSON": "I-PER",
              "LOCATION": "I-LOC",
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
    OUTPUT_FILE = "stanford_output.txt"
    text = conll2string(INPUT_FILE)
    ners = ner_stanford(text)
    out = output_format(INPUT_FILE,ners)
    print(out)
    #write2file(out,OUTPUT_FILE)

