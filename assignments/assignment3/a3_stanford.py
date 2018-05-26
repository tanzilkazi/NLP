# -*- coding: utf-8 -*-

import pandas as pd
import os
import nltk as nl
from nltk.tokenize import word_tokenize

STANFORD_NER = r".\stanford\stanford-ner-2018-02-27\stanford-ner.jar"
STANFORD_POS = r".\stanford\stanford-postagger-full-2018-02-27\stanford-postagger.jar"
STANFORD_PARSER = r".\stanford\stanford-parser-full-2018-02-27\stanford-parser.jar"
os.environ['JAVAHOME'] = r"C:\Program Files\Java\jre1.8.0_91\bin\java.exe"
os.environ['CLASSPATH'] = STANFORD_NER;STANFORD_POS;STANFORD_PARSER

st_ner = nl.tag.StanfordNERTagger(r".\stanford\stanford-english-corenlp-2018-02-27-models\edu\stanford\nlp\models\ner\english.conll.4class.distsim.crf.ser.gz")

st_pos = nl.tag.StanfordPOSTagger()


text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st_ner.tag(tokenized_text)



print(classified_text)