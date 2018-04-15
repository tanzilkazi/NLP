# COMP5046: Natural Language Processing
# Assignment 1: Annotation Task
# by Tanzil Kazi (ID: 470239029)

import pandas as pd
from collections import Counter
from sklearn import metrics as mets
import seaborn as sb
import matplotlib.pyplot as plot
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import file from CSV to dataframe
anno_file = pd.read_csv("annotations.csv")
my_userID = "tkaz8954"

# Extract unique IDs (IDs are titles of articles), categories and userIDs
IDs = list(set(anno_file['ID']))
labels = list(set(anno_file['Annotation']))
userIDs = list(set(anno_file['Annotator']))

# voted_results consist of annotations' IDs and their true categories
voted_results = {}
predictions = {}

# Apply groupby to group Annotators and count the number of annotations, select annotation not equal to 80
valid_annotator = anno_file.groupby(['Annotator']).count()['Annotation']
drop_annotators = valid_annotator[valid_annotator != 80].index.values

# Drop the annotators and create a new copy of the dataframe
for x in drop_annotators:
    drop_indexes = anno_file.loc[anno_file['Annotator'] == x].index.values
    anno_file = anno_file.drop(drop_indexes)

for ID in IDs:
    # Slices the dataframe for each title and counts the Annotation column
    label_counts = Counter(anno_file.loc[anno_file['ID'] == ID]['Annotation'])
    # Extract the most common label (which appears in the first element) and also extract it's name
    true_label = label_counts.most_common()[0][0]
    voted_results[ID] = true_label

for index,row in anno_file.loc[anno_file['Annotator'] == my_userID].iterrows():
    predictions[row['ID']] = row['Annotation']

# Extract voted and predicted categories into the arrays - the arrays are aligned
voted_category = []
predicted_category = []
for key, value in voted_results.items():
    voted_category.append(value)
    predicted_category.append(predictions[key])

# Confusion Matrix
conf_matrix = mets.confusion_matrix(voted_category, predicted_category, labels=labels)
df_conf_matrix = pd.DataFrame(conf_matrix,index=labels,columns=labels)

fig = plot.figure(figsize=(6,6))
conf_map = sb.heatmap(df_conf_matrix, annot=True,cmap="Blues",linewidths=0.5,linecolor="black",cbar=False,square=True)
plot.xticks(rotation=90)
plot.yticks(rotation=0)
plot.xlabel("Predicted Category")
plot.ylabel("Voted Category")

#Precision, recall, f1_score
class_rpt = mets.classification_report(voted_category, predicted_category)
print(class_rpt)

# Precision
precision = mets.precision_score(voted_category, predicted_category, average='weighted', labels=labels)
print("Average(weighted) precision is {0:.2f}".format(precision))

# Recall
recall = mets.recall_score(voted_category, predicted_category, average='weighted', labels=labels)
print("Average(weighted) recall is {0:.2f}".format(recall))

# F1-score
f_score = mets.recall_score(voted_category, predicted_category, average='weighted', labels=labels)
print("Average(weighted) F1-score is {0:.2f}".format(f_score))

# Cohen's Kappa
ckappa = mets.cohen_kappa_score(voted_category, predicted_category, labels=labels)
print("\nCohen's Kappa is {0:.2f}".format(ckappa))

# Fleiss Kappa
# Create an index for the IDs - IDs are strings and can't be used as index in dataframe. The indexes are 0-79.
ID_index = {}
ID_index_counter = 0
for ID in IDs:
    ID_index[ID] = ID_index_counter
    ID_index_counter = ID_index_counter + 1

# Create empty dataframe to store
df_fkappa = pd.DataFrame(columns=labels,index=list(range(len(IDs)))).fillna(0)

# For each subject (ID), extract all annotations and count them. Then insert the counts in dataframe
for ID in IDs:
    annotation = anno_file.loc[anno_file['ID'] == ID]['Annotation']
    annotation_counts = annotation.value_counts()
    for label, count in annotation_counts.items():
        df_fkappa[label][ID_index.get(ID)] = count

# Calculating Fleiss' Kappa as per https://en.wikipedia.org/wiki/Fleiss%27_kappa#Worked_example
sum_all_cells = df_fkappa.sum(axis=1).sum()
p = df_fkappa.sum(axis=0)/sum_all_cells

df_fkappa_sq = np.square(df_fkappa)
n_annotators = len(userIDs) - len(drop_annotators)
P = (df_fkappa_sq.sum(axis=1) - n_annotators)/(n_annotators*(n_annotators-1))

Pbar = sum(P)/len(IDs)
Pbar_e = np.sum(np.square(p))

fkappa = (Pbar-Pbar_e)/(1-Pbar_e)
print("\nFleiss' Kappa is {0:.2f}".format(fkappa))

# Display confusion matrix
plot.show()