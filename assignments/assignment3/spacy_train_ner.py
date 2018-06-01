#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
import time

# training data
# TRAIN_DATA = [
#     ('Who is Shaka Khan?', {
#         'entities': [(7, 17, 'PERSON')]
#     }),
#     ('I like London and Berlin.', {
#         'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
#     })
# ]

TEST_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=r".\spacy_model", n_iter=50):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            time_start = time.time()
            for text, annotations in TRAIN_DATA:
                if "GOLF" in text:
                    print("")
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses, time.time()-time_start)

    # test the trained model
    for text, _ in TEST_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TEST_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])



def conll2spacytrain(file):
    BREAK_COUNT = 500
    corpus = []
    line_text = ""
    ent_list = []
    count = 0
    tag = "O\n"
    new_doc_tag = False
    with open(INPUT_FILE) as file:
        for line in file:
            conll_line = line.split(sep=" ")
            if conll_line[0] == '\n':
                continue
            elif conll_line[0] == '-DOCSTART-' and new_doc_tag == True:
                new_doc_tag = False
                line_text = line_text.strip()
                ent_dict = {'entities': ent_list}
                corpus.append((line_text, ent_dict))
                line_text = ""
                ent_list = []
                new_doc_tag = True
            elif conll_line[0] != '-DOCSTART-' and new_doc_tag == True:
                word = conll_line[0]
                if len(conll_line) > 1 and conll_line[3] != tag:
                    start_len = len(line_text)
                    end_len = start_len+len(word)
                    ner_tag = conll_line[3]
                    ent_list.append((start_len, end_len, ner_tag.replace("\n", "")))
                word = word.replace(r"'","|")
                line_text = line_text + " " + word.strip()
            elif conll_line[0] == '-DOCSTART-':
                new_doc_tag = True

            else:
                print("DO NOTHING\n")
            # if len(line) == 1: # newline
            #     line_text = line_text.strip()
            #     ent_dict = {'entities':ent_list}
            #     corpus.append((line_text,ent_dict))
            #     line_text = ""
            #     ent_list = []
            # else:
            #     conll_line = line.split(sep=" ")
            #     if conll_line[3] != tag:
            #         start_len = len(line_text)
            #         end_len = start_len+len(conll_line[0])
            #         ner_tag = conll_line[3]
            #
            #         ent_list.append((start_len,end_len,ner_tag.replace("\n","")))
            #     line_text = line_text + " " + line.split(sep=" ")[0]
            count = count + 1
            # if count == BREAK_COUNT:
            #     break
    return corpus

if __name__ == '__main__':
    INPUT_FILE = r".\conll03\eng.train"
    TRAIN_DATA = conll2spacytrain(INPUT_FILE)
    #print(TRAIN_DATA)
    with open("TRAIN_DATA.txt","w") as f:
        f.write(str(TRAIN_DATA))
    plac.call(main)