from anago.utils import download
import anago
import pandas as pd
from sklearn import metrics
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

data = open('eng.testb')
array = []

dir_path = 'models'
# url = 'https://storage.googleapis.com/chakki/datasets/public/models.zip'
# download(url, dir_path)
model = anago.Sequence.load(dir_path)

# x_train, y_train = read.load_data_and_labels('train.txt')
# x_valid, y_valid = read.load_data_and_labels('valid.txt')
# x_test, y_test = read.load_data_and_labels('test.txt')
#
#
# model = anago.Sequence()
# model.train(x_train, y_train, x_valid, y_valid)


for lines in data:
   if len(lines) != 1:  # newline
       array.append(lines.split())


df = pd.DataFrame(array, columns=['Tokens', 'POS-TAG', 'IOB-POS', 'IOB-NER'])
tokens = df.iloc[:,0].str.cat(sep=' ').split()

tags = model.analyze(tokens)

predict = ['O']*len(tokens)

for entities in tags['entities']:
    for i in range(entities['beginOffset'], entities['endOffset']):
        predict[i] = "I-"+entities['type']


newdf = pd.DataFrame(predict, columns=['NER-PRED'])
frames = [df, newdf]
result = pd.concat(frames, axis = 1)

np.savetxt(r'C:\Users\azadt\Documents\NLP\untitled\output.txt', result.values, fmt='%s')


Y = result['IOB-NER'].tolist()
Y_pred = result['NER-PRED'].tolist()

final_Y = []
final_preds = []

for i in range(0,len(Y)):
    if Y[i] != 'O':
        final_Y.append(Y[i][2:])
        if Y_pred[i] !='O':
            final_preds.append(Y_pred[i][2:])
        else:
            final_preds.append(Y_pred[i])

class_rpt = metrics.classification_report(final_Y, final_preds)

print(class_rpt)

labels = list(set(final_preds))

confusion = metrics.confusion_matrix(final_Y, final_preds, labels=labels)
df_conf_matrix = pd.DataFrame(confusion,index=labels,columns=labels)

fig = plt.figure(figsize=(8,8))
sb.set(font_scale=0.8)
heatmap = sb.heatmap(df_conf_matrix, square=True, linewidth=0.25, annot=True, cbar=False, cmap="Greens", fmt='g')
heatmap.invert_yaxis()
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xlabel("Predicted Category")
plt.ylabel("Gold Category")
plt.show()