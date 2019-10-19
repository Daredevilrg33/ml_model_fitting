from scipy.io import arff
import os

try:
	import pandas as pd
except ImportError:
	print("installing pandas---->")
	os.system("conda install pandas")
	print("installation complete---->")
	import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np



DEFAULT_KNN_NEIGHBOURS = 10
TEST_SIZE_SPLIT = 0.4
CV_FOLD = 10


def get_data(file_path):
	'''
	This method is used to load the arff data and seperate out X and y labels
	'''
	data = arff.loadarff(file_path)
	dataset = pd.DataFrame(data[0])
	X = dataset.iloc[:, 0: 19].values
	y = dataset.iloc[:, 19].values.astype('int') # convert to int from object type
	return X,y

def get_train_and_test_split(X, y, test_size=TEST_SIZE_SPLIT, stratify=None):
	'''
	Splits the dataset based on the test_size value provided
	Returns X_train, y_train, X_test, y_test as tuples
	'''
	if not stratify:
		stratify = y
	return train_test_split(X, y, test_size=test_size, random_state=0, stratify=stratify)

def knn_classifier(X, y, X_test, neighbors=DEFAULT_KNN_NEIGHBOURS):
	'''
	Creates an object of knn, fits training data, predict X_test and returns
	the instance so we can do other stuff with it like plotting graph or calculating score 
	'''
	knn = KNeighborsClassifier(n_neighbors=neighbors)
	knn.fit(X,y)
	knn.predict(X_test)
	return knn

def knn_score(X, X_test, y, y_test, neighbors=DEFAULT_KNN_NEIGHBOURS):
	'''
	Returns the score of knn by fitting training data
	'''
	knn = knn_classifier(X, y, X_test)
	return knn.score(X_test, y_test)

def get_kfold_cross_validation(classifier, X, y, k_fold=CV_FOLD):
	'''
	Returns the mean score of knn by doing cross validation
	'''
	cv_scores = cross_val_score(classifier, X, y, cv=k_fold)
	return np.mean(cv_scores)


def grid_search_with_cross_validation(param_grid, classifier, X, y):
	'''
	Tries to find optimal value of paramters for a model by using cross validations
	'''
	classifier_gscv = GridSearchCV(classifier, param_grid, cv=CV_FOLD)
	classifier_gscv.fit(X, y)
	return classifier_gscv

def __main__():
	X, y = get_data('./data/messidor_features.arff')
	print('----- Knn score: {} -----'.format(knn_score(*get_train_and_test_split(X,y))))
	knn_cv = KNeighborsClassifier(n_neighbors=DEFAULT_KNN_NEIGHBOURS)
	print('----- Knn score with cross validation: {} -----'.format(get_kfold_cross_validation(knn_cv, X, y)))
	_knn_gscv = KNeighborsClassifier()
	knn_gscv = grid_search_with_cross_validation({'n_neighbors': np.arange(1, 30)}, _knn_gscv, X, y)
	print('----- Knn best value for k: {} with score: {} -----'.format(knn_gscv.best_params_, knn_gscv.best_score_))

__main__()