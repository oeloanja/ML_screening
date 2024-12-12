import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV

import pandas as pd

data = pd.read_csv('./data_fin.csv')

Y = data[['target']]
X = data.drop('target', axis = 1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(test_x)
print('accuracy : {0:.4f}'.format(accuracy_score(test_y, pred)))
print('f1-score : {0:.4f}'.format(f1_score(test_y, pred, average='macro')))
print('precision : {0:.4f}'.format(precision_score(test_y, pred, average='macro')))
print('recall : {0:.4f}'.format(recall_score(test_y, pred, average='macro')))