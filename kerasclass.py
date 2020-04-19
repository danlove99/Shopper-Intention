import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

class model:
	def __init__(self, input_size):
		self.nn = Sequential()
		self.nn.add(Dense(16, input_dim=input_size, activation='relu'))
		self.nn.add(Dropout(0.4))
		self.nn.add(Dense(32, activation='relu'))
		self.nn.add(Dropout(0.4))
		self.nn.add(Dense(64, activation='relu'))
		self.nn.add(Dropout(0.4))
		self.nn.add(Dense(32, activation='relu'))
		self.nn.add(Dropout(0.4))
		self.nn.add(Dense(16, activation='relu'))
		self.nn.add(Dropout(0.4))
		self.nn.add(Dense(8, activation='relu'))
		self.nn.add(Dropout(0.4))
		self.nn.add(Dense(1, activation='sigmoid'))
		self.nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	def train(self, X, y):
		self.nn.fit(X,y, epochs=5, batch_size=10)