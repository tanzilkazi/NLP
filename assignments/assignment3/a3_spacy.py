import os

BREAK_COUNT = 250

def conll2spacytrain(file):
    corpus = []
    line_text = ""
    ent_list = []
    count = 0
    tag = "O\n"
    with open(INPUT_FILE) as file:
        for line in file:
            if len(line) == 1: # newline
                line_text = line_text.strip()
                ent_dict = {'entities':ent_list}
                corpus.append((line_text,ent_dict))
                line_text = ""
                ent_list = []
            else:
                conll_line = line.split(sep=" ")
                if conll_line[3] != tag:
                    start_len = len(line_text)
                    end_len = start_len+len(conll_line[0])
                    ner_tag = conll_line[3]

                    ent_list.append((start_len,end_len,ner_tag.replace("\n","")))
                line_text = line_text + " " + line.split(sep=" ")[0]
            count = count + 1
            if count == BREAK_COUNT:
                break
    return corpus

if __name__ == "__main__":
    INPUT_FILE = r".\conll03\eng.testb"
    OUTPUT_FILE = "spacy_train.txt"
    text = conll2spacytrain(INPUT_FILE)
    print(text)
    # with open(OUTPUT_FILE,"w") as f:
    #     f.write(text)