import pandas as pd
from collections import Counter
from sklearn import metrics as mets
import seaborn as sb
import matplotlib.pyplot as plot
import numpy as np

# import file from CSV to dataframe
anno_file = pd.read_csv("annotations.csv")
my_userID = "taza9236"

# extract unique IDs - IDs are titles of articles, labels and all userIDs
IDs = list(set(anno_file['ID']))
labels = list(set(anno_file['Annotation']))
userIDs = list(set(anno_file['Annotator']))

# gold_labels consist of annotations' IDs and their true labels
gold_results = {}
predictions = {}

# apply groupby to group Annotators and count the number of annotations, select annotation not equal to 80
valid_annotator = anno_file.groupby(['Annotator']).count()['Annotation']
drop_annotators = valid_annotator[valid_annotator != 80].index.values

# drop the annotators and create a new copy of the dataframe
for x in drop_annotators:
    drop_indexes = anno_file.loc[anno_file['Annotator'] == x].index.values
    anno_file = anno_file.drop(drop_indexes)

for ID in IDs:
    # Slices the dataframe for each title and counts the Annotation column
    label_counts = Counter(anno_file.loc[anno_file['ID'] == ID]['Annotation'])
    # Extract the most common label (which appears in the first element) and also extract it's name
    true_label = label_counts.most_common()[0][0]
    gold_results[ID] = true_label

for index,row in anno_file.loc[anno_file['Annotator'] == my_userID].iterrows():
    predictions[row['ID']] = row['Annotation']

# extract gold and predicted labels into arrays which are aligned
gold_labels = []
predicted_labels = []
for key, value in gold_results.items():
    gold_labels.append(value)
    predicted_labels.append(predictions[key])

# Confusion Matrix
conf_matrix = mets.confusion_matrix(gold_labels,predicted_labels, labels=labels)
df_conf_matrix = pd.DataFrame(conf_matrix,index=labels,columns=labels)

fig = plot.figure(figsize=(6,6))
#conf_map = sb.heatmap(df_conf_matrix, annot=True,cmap="Blues",linewidths=0.5,linecolor="black",cbar=False,square=True)
sb.set(font_scale=1.4)
heatmap = sb.heatmap(df_conf_matrix, square=True, linewidth=0.25, annot=True, cbar=False, cmap="Greens")
heatmap.invert_yaxis()
plot.xticks(rotation=90)
plot.yticks(rotation=0)
plot.xlabel("Predicted Category")
plot.ylabel("Gold Category")
plot.show()




#Precision, recall, f1_score
class_rpt = mets.classification_report(gold_labels,predicted_labels)
print(class_rpt)

# Precision
precision = mets.precision_score(gold_labels,predicted_labels,average='weighted', labels=labels)
print("Overall precision is {0:.2f}".format(precision))

# Recall
recall = mets.recall_score(gold_labels,predicted_labels,average='weighted', labels=labels)
print("Overall recall is {0:.2f}".format(recall))

# F1-score
f_score = mets.recall_score(gold_labels,predicted_labels,average='weighted', labels=labels)
print("Overall F1 score is {0:.2f}".format(f_score))

# Cohen's Kappa
ckappa = mets.cohen_kappa_score(gold_labels,predicted_labels, labels=labels)
print("Cohen's Kappa is {0:.2f}".format(ckappa))

# Fleiss Kappa
# Create an index for the IDs - IDs are strings and can't be used as index in dataframe. The indexes are 0-79.
ID_index = {}
ID_index_counter = 0
for ID in IDs:
    ID_index[ID] = ID_index_counter
    ID_index_counter = ID_index_counter + 1



#Create empty dataframe to store
df_fkappa = pd.DataFrame(columns=labels,index=list(range(len(IDs)))).fillna(0)

# for each subject (ID), extract all annotations and count them. Then insert the counts in dataframe
for ID in IDs:
    annotation = anno_file.loc[anno_file['ID'] == ID]['Annotation']
    annotation_counts = annotation.value_counts()
    for label, count in annotation_counts.items():
        df_fkappa[label][ID_index.get(ID)] = count

# calculating Fleiss Kappa
sum_all_cells = df_fkappa.sum(axis=1).sum()
p = df_fkappa.sum(axis=0)/sum_all_cells

df_fkappa_sq = np.square(df_fkappa)
n_annotators = len(userIDs) - len(drop_annotators)
P = (df_fkappa_sq.sum(axis=1) - n_annotators)/(n_annotators*(n_annotators-1))

Pbar = sum(P)/len(IDs)
Pbar_e = np.sum(np.square(p))

fkappa = (Pbar-Pbar_e)/(1-Pbar_e)
print("Fleiss' Kappa is {0:.2f}".format(fkappa))