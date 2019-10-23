from sklearn.neighbors import KNeighborsClassifier
import numpy as np

DEFAULT_KNN_NEIGHBORS = 10

class KnnClassifier():
	
	def __init__(self):
		self.knn = KNeighborsClassifier(n_neighbors=DEFAULT_KNN_NEIGHBORS)
		print("""
			**********************
			KNN
			**********************
		""")
	
	def train_and_predict(self, X, y, X_test):
		'''
		fit training dataset and predict values for test dataset 
		'''
		self.knn.fit(X,y)
		self.knn.predict(X_test)

	def score(self, X, X_test, y, y_test):
		'''
		Returns the score of knn by fitting training data
		'''
		self.train_and_predict(X, y, X_test)
		return self.knn.score(X_test, y_test)

	def create_new_instance(self, values, with_default_values=True):
		if with_default_values:
			return KNeighborsClassifier(n_neighbors=DEFAULT_KNN_NEIGHBORS)
		else:
			return KNeighborsClassifier(**values)

	def param_grid(self):
		'''
		dictionary of hyper-parameters to get good values for each one of them
		'''
		return {'n_neighbors': np.arange(1, 30)}

	def __str__(self):
		return "KNN"