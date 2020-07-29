import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.estimator import regression

class model:
	def __init__(self, input_size):
		self.nn = input_data(shape=[None, input_size, 1], name='input')
		self.nn = fully_connected(self.nn,16, activation='relu')
		self.nn = dropout(self.nn, 0.4)
		self.nn = fully_connected(self.nn,32, activation='relu')
		self.nn = fully_connected(self.nn,64, activation='relu')
		self.nn = fully_connected(self.nn,32, activation='relu')
		self.nn = dropout(self.nn, 0.4)
		self.nn = fully_connected(self.nn,16, activation='relu')
		self.nn = fully_connected(self.nn, 1, activation='sigmoid')
		self.nn = regression(self.nn, optimizer='adam', learning_rate=1e-3, loss='binary_crossentropy', name='targets')
		self.nn = tflearn.DNN(self.nn, tensorboard_dir='log')

	def train(self, X, y):
		self.nn.fit(X,y, n_epoch=10, snapshot_step=200, show_metric=True, batch_size=10)
