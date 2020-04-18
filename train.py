from numpy import loadtxt, reshape, expand_dims
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.estimator import regression
import pandas as pd
from sklearn import preprocessing
from  sklearn.model_selection import train_test_split
import urllib.request

# load the dataset
url = 'online_shoppers_intention.csv'
df = pd.read_csv(url)
y = df['Revenue'].values

X = df.drop(['Revenue'], axis=1)

# Normalize data
X =((X-X.min())/(X.max()-X.min()))*20

# split and reshape data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X = expand_dims(X_train, axis=2)
y.shape = (12330, 1)
X = np.nan_to_num(X)


import tflearnclass

model = tflearnclass.model(17)
model.train(X, y)

# evaluate the model

precition = model.predict(X[0])
print(y[0])