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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

TEST_SIZE_SPLIT = 0.4
CV_FOLD = 10


class Model():

	def __init__(self, model_type):
		self.model_type = model_type

	def get_data(self, file_path):
		if file_path.endswith('arff'):
			self.get_arff_data(file_path)
		else:
			print("Don't know how to load this data")

	def get_arff_data(self, file_path):
		'''
		This method is used to load the arff data and seperate out X and y labels
		'''
		data = arff.loadarff(file_path)
		dataset = pd.DataFrame(data[0])
		self.X = dataset.iloc[:, 0: 19].values
		self.y = dataset.iloc[:, 19].values.astype('int') # convert to int from object type	

	def get_train_and_test_split(self, test_size=TEST_SIZE_SPLIT, stratify=True):
		'''
		Splits the dataset based on the test_size value provided
		sets X_train, X_test, y_train, y_test values for the instance
		'''
		if stratify:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0, stratify=self.y)
		else:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)

	def get_score_without_any_processing(self):
		'''
			Delegates the score method to individual model and gets the score
		'''
		self.get_train_and_test_split()
		score = self.model_type.score(self.X_train, self.X_test, self.y_train, self.y_test)
		print('----- {} score: {} -----'.format(self.model_type, score))

	def get_kfold_cross_validation(self, k_fold=CV_FOLD):
		'''
		Returns the mean score of knn by doing cross validation
		'''
		classifier = self.model_type.create_new_instance(with_default_values=True)
		cv_scores = cross_val_score(classifier, self.X, self.y, cv=k_fold)
		print('----- {} score with cross validation: {} -----'.format(self.model_type, np.mean(cv_scores)))

	def grid_search_with_cross_validation(self):
		'''
		Tries to find optimal value of paramters for a model by using cross validations
		'''
		classifier = self.model_type.create_new_instance(with_default_values=False)
		classifier_gscv = GridSearchCV(classifier, self.model_type.param_grid(), cv=CV_FOLD)
		classifier_gscv.fit(self.X, self.y)
		print('----- {} best value for k: {} with score: {} -----'.format(self.model_type, classifier_gscv.best_params_, classifier_gscv.best_score_))
