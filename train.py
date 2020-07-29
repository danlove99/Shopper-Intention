from numpy import loadtxt, reshape, expand_dims
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
import os 

# Attempt to load numpy array else load the dataset
try:
	X = np.load('features.npy')
	y = np.load('labels.npy')
	print("Succesfully loaded preprocessed features and labels!")
except:
	# load the dataset
	url = 'online_shoppers_intention.csv'
	df = pd.read_csv(url)
	y = df['Revenue'].values

	X = df.drop(['Revenue'], axis=1)

	# Normalize data
	X =((X-X.min())/(X.max()-X.min()))*20
	np.save('features.npy', X)
	np.save('labels.npy', y)

# split and reshape data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

choice = input('k = Keras | t = TFLearn \nWhat framework do you want to use: ')
# if using tflearn, make data suitable
if choice.lower() == 't':
	X = expand_dims(X_train, axis=2)

y.shape = (12330, 1)
X = np.nan_to_num(X) # Convert all instances of NaN to 0

if choice.lower() == 'k': # Keras
	import kerasclass

	model = kerasclass.model(17)
	model.train(X,y)
elif choice.lower() == 't': # TFLearn
	import tflearnclass

	model = tflearnclass.model(17)
	model.train(X, y)
else:
	print('Not an option! Please enter either "k" or "t"')
	
