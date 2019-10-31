from sklearn.neural_network import MLPClassifier
import numpy as np
import random

DEFAULT_ITERATIONS = 500
HIDDEN_LAYER_SIZES = (20, 18)


class MlpClassifier():
	
	def __init__(self):
		self.nn = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=DEFAULT_ITERATIONS, random_state=0)
		print("""
			**********************
			Neural Network
			**********************
		""")
	
	def train_and_predict(self, X, y, X_test):
		'''
		fit training dataset and predict values for test dataset 
		'''
		self.nn.fit(X,y)
		self.nn.predict(X_test)

	def score(self, X, X_test, y, y_test):
		'''
		Returns the score of knn by fitting training data
		'''
		self.train_and_predict(X, y, X_test)
		return self.nn.score(X_test, y_test)

	def create_new_instance(self, values, with_default_values=True):
		if with_default_values:
			return MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=DEFAULT_ITERATIONS, random_state=0)
		else:
			return MLPClassifier(**{**values, 'random_state': 0})

	def param_grid(self, is_random=False):
		'''
		dictionary of hyper-parameters to get good values for each one of them
		'''
		# random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
		return {
			'hidden_layer_sizes': [
				(random.randrange(1, 20), random.randrange(1, 20)), 
				(random.randrange(1, 20), random.randrange(1, 20))
			], 'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001]
    }

	def __str__(self):
		return "Neural Network"