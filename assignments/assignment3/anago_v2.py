import anago
from anago.utils import download
from anago.utils import load_data_and_labels
import string as st

def conll2string(file):
    corpus = ""
    line_text = ""
    count = 0
    new_doc_tag = False
    with open(INPUT_FILE) as file:
        for line in file:
            conll_line = line.split(sep=" ")
            if conll_line[0] == '\n':
                continue
            elif conll_line[0] == '-DOCSTART-' and new_doc_tag == True:
                new_doc_tag = False
                line_text = line_text.strip()
                corpus = corpus + " " + line_text + "\n"+ '-DOCSTART-' + "\n"
                line_text = ""
                new_doc_tag = True
            elif conll_line[0] != '-DOCSTART-' and new_doc_tag == True:
                word = conll_line[0]
                line_text = line_text + " " + word.strip()
            elif conll_line[0] == '-DOCSTART-':
                line_text = line_text + '-DOCSTART-' + "\n"
                new_doc_tag = True
            else:
                print("DO NOTHING\n")
            count = count + 1
        new_doc_tag = False
        line_text = line_text.strip()
        corpus = corpus + " " + line_text + "\n"
        line_text = ""
    return corpus

def download_trained_anago():
    # downloads 'weights.h5', 'params.json', 'preprocessor.pickle' in local folder
    url = 'https://storage.googleapis.com/chakki/datasets/public/ner/model_en.zip'
    download(url)
    return True

def predict_ners(text):
    model = anago.Sequence.load('weights.h5', 'params.json', 'preprocessor.pickle')
    corpus = ""
    for line in text.split("\n"):
        tokens = line.strip().split(sep=" ")
        corpus = corpus  + ' '.join(str(e) for e in tokens)
    corpus = corpus.replace("-DOCSTART-", " ")
    return model.analyze(corpus)["entities"]

def nltk2conll_mapper(data):
    mapper = {"MISC": "I-MISC",
              "ORG": "I-ORG",
              "PER": "I-PER",
              "LOC": "I-LOC",
              "O": "O"
              }
    index = 0
    while index < len(data):
        if(data[index][-1] in mapper):
            data[index][-1] = mapper[data[index][-1]]
        index = index + 1
    return data

def output_format(in_file,named_ents):
    output = []
    ner_count = 0
    file = open(in_file, "r")
    lines = file.readlines()
    file.close()
    nline = 0
    while nline < len(lines):
        line = lines[nline]
        out = []
        is_ner = False
        if line != '\n':
            line = line.strip("\n")
            out = line.split(sep=" ")
            if (ner_count < len(named_ents)):
                ner_w_count = named_ents[ner_count]["endOffset"] - named_ents[ner_count]["beginOffset"]
                if (ner_w_count == 1):
                    if out[0] == named_ents[ner_count]["text"]:
                        out.append(named_ents[ner_count]["type"])
                        output.append(out)
                        ner_count = ner_count + 1
                        nline = nline + 1
                        is_ner = True
                    else:
                        nline = nline + 1
                else:
                    list_ner = named_ents[ner_count]["text"].split(sep=" ")
                    if out[0] == list_ner[0]:
                        for ner_word in list_ner:
                            out.append(named_ents[ner_count]["type"])
                            output.append(out)
                            nline = nline + 1
                            out = lines[nline].strip("\n").split(sep=" ")
                        ner_count = ner_count + 1
                        is_ner = True
                    else:
                        nline = nline + 1
                if not is_ner:
                    out.append("O")
                    output.append(out)
            else:
                nline = nline + 1
        else:
            out.append("\n")
            output.append(out)
            nline = nline + 1
    return nltk2conll_mapper(output)

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

if __name__ == "__main__":

    INPUT_FILE = r".\conll03\eng.testb"
    OUTPUT_FILE = "anago_output.txt"
    # download_trained_anago()
    text = conll2string(INPUT_FILE)
    ners = predict_ners(text)
    out = output_format(INPUT_FILE,ners)
    write2file(out,OUTPUT_FILE)





