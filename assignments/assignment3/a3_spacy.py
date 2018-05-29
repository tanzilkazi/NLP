import os

BREAK_COUNT = 20

def conll2spacytrain(file):
    corpus_text = ""
    line_text = ""
    ent_list = []
    count = 0
    tag = "O\n"
    with open(INPUT_FILE) as file:
        for line in file:
            if len(line) == 1: # newline
                line_text = line_text.strip()
                str_ent_list = str(ent_list).strip()
                str_ent_list = str_ent_list.replace(r"\\","").strip()
                corpus_text = corpus_text + "(\"" + line_text + "\", {'entities':"+str_ent_list+ "}) "+ '\n'
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
    corpus_text = "["+corpus_text+"]"
    return corpus_text

if __name__ == "__main__":
    INPUT_FILE = r".\conll03\eng.testb"
    OUTPUT_FILE = "spacy_train.txt"
    text = conll2spacytrain(INPUT_FILE)
    with open(OUTPUT_FILE,"w") as f:
        f.write(text)