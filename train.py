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

# Build the model
nn = input_data(shape=[None, 17, 1], name='input')
nn = fully_connected(nn,16, activation='relu')
nn = dropout(nn, 0.4)
nn = fully_connected(nn,32, activation='relu')
nn = dropout(nn, 0.4)
nn = fully_connected(nn,64, activation='relu')
nn = dropout(nn, 0.4)
nn = fully_connected(nn,32, activation='relu')
nn = dropout(nn, 0.4)
nn = fully_connected(nn,16, activation='relu')
nn = dropout(nn, 0.4)
nn = fully_connected(nn, 1, activation='sigmoid')
nn = regression(nn, optimizer='adam', learning_rate=1e-3, loss='binary_crossentropy', name='targets')
nn = tflearn.DNN(nn, tensorboard_dir='log')

# fit the model
nn.fit(X, y, n_epoch=10, snapshot_step=200, show_metric=True, batch_size=10)

# evaluate the model
accuracy = nn.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy[0]*100))
