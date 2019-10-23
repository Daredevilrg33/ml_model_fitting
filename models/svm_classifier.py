from sklearn.svm import SVC
import numpy as np

DEFAULT_KERNEL = 'rbf'
DEFAULT_C = 1.0

class SvmClassifier():
	
	def __init__(self):
		self.svm = SVC(kernel=DEFAULT_KERNEL, C=DEFAULT_C, gamma='auto', random_state=0)
		print("""
			**********************
			SVM
			**********************
		""")
	
	def train_and_predict(self, X, y, X_test):
		'''
		fit training dataset and predict values for test dataset 
		'''
		self.svm.fit(X,y)
		self.svm.predict(X_test)

	def score(self, X, X_test, y, y_test):
		'''
		Returns the score of knn by fitting training data
		'''
		self.train_and_predict(X, y, X_test)
		return self.svm.score(X_test, y_test)

	def create_new_instance(self, values, with_default_values=True):
		if with_default_values:
			return SVC(kernel=DEFAULT_KERNEL, C=DEFAULT_C, gamma='auto', random_state=0)
		else:
			return SVC(**{**values, 'random_state': 0})

	def param_grid(self):
		'''
		dictionary of hyper-parameters to get good values for each one of them
		'''
		return [
			# {'C': [1, 10, 100, 1000], 'kernel': ['linear','poly']}, 
			{'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]},
		]

	def __str__(self):
		return "SVM"