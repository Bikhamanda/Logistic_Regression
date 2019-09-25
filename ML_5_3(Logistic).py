import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris

data = load_iris()
print(dir(data))
df = pd.DataFrame(
    data['data'],
    columns = ['SL', 'SW', 'PL', 'PW']
)
df['target'] = data['target']
df['spesies'] = df['target'].apply(
    lambda x: data['target_names'][x]
)
print(df['spesies'])
# print(df.head())

# split data = 5% test
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    df[['SL', 'SW', 'PL', 'PW']],
    df['target'],
    test_size = .5
)
# print(len(xtr))
# print(len(ytr))


# Logistic reg
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver= 'lbfgs')
model.fit(xtr, ytr)

# print(xts)
# print(yts)
# print(model.predict(xts))

# print(model.predict([[ 9,9,9,9 ]]))
# print(model.predict_proba([[ 9,9,9,9 ]]))
# print(model.score(xtr, ytr))