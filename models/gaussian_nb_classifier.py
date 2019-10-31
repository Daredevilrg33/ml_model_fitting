from sklearn.naive_bayes import GaussianNB
import numpy as np

class GaussianNbClassifier():
	
	def __init__(self):
		self.nb = GaussianNB()
		print("""
			**********************
			GaussianNB
			**********************
		""")
	
	def train_and_predict(self, X, y, X_test):
		'''
		fit training dataset and predict values for test dataset 
		'''
		self.nb.fit(X,y)
		self.nb.predict(X_test)

	def score(self, X, X_test, y, y_test):
		'''
		Returns the score of knn by fitting training data
		'''
		self.train_and_predict(X, y, X_test)
		return self.nb.score(X_test, y_test)

	def create_new_instance(self, values, with_default_values=True):
		if with_default_values:
			return GaussianNB()
		else:
			return GaussianNB(**{**values})

	def param_grid(self, is_random=False):
		'''
		dictionary of hyper-parameters to get good values for each one of them
		'''
		# random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
		return {}

	def __str__(self):
		return "GaussianNB"