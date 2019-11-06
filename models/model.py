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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

TEST_SIZE_SPLIT = 0.3
CV_FOLD = 7

class Model():

	def __init__(self, model_type):
		self.model_type = model_type

	def get_data(self, file_path):
		if file_path.endswith('arff'):
			self.get_arff_data(file_path)
		elif file_path.endswith('data'):
			if file_path.endswith('wisconsin.data'):
				self.get_bc_wisconsin_data(file_path)
			elif 'german' in file_path:
				self.get_german_credit_data(file_path)
			else:
				self.get_au_credit_data(file_path)
		elif file_path.endswith('NNA'):
			self.get_steel_plates_faults_data(file_path)
		else:
			print("Don't know how to load this data")

	def get_steel_plates_faults_data(self, file_path):
		'''
		This method loads the data and converts the available data and assign classes as follows
		0 - Pastry, 1 - Z_Scratch, 2 - K_Scatch, 3 - Stains, 4 - Dirtiness, 5 - Bumps, 6 - Other_Faults
		'''
		data = pd.read_csv(file_path, header=None, delim_whitespace=True)
		y_all = np.array(data.iloc[:, 27:])
		y = np.empty(1941, dtype=np.int)
		y[y_all[:,0] == 1] = 0
		y[y_all[:,1] == 1] = 1
		y[y_all[:,2] == 1] = 2
		y[y_all[:,3] == 1] = 3
		y[y_all[:,4] == 1] = 4
		y[y_all[:,5] == 1] = 5
		y[y_all[:,6] == 1] = 6
		self.y = y
		self.X = np.array(data.iloc[:,:27])

	def get_german_credit_data(self, file_path):
		'''
		This method is used to load the data from a file which has .data extension and seperate out X and y labels
		'''
		data = pd.read_csv(file_path, header=None, delim_whitespace=True)
		y = np.array(data[24])
		self.y = y.astype('int')
		self.X = np.array(data.iloc[:, :24])

	def get_au_credit_data(self, file_path):
		'''
		This method is used to load the data from a file which has .data extension and seperate out X and y labels
		'''
		data = pd.read_csv(file_path, header=None, delim_whitespace=True)
		y = np.array(data[14])
		self.y = y.astype('int')
		self.X = np.array(data.iloc[:, :14])

	def get_bc_wisconsin_data(self, file_path):
		'''
		This method is used to load the data from a file which has .data extension and seperate out X and y labels
		'''
		data = pd.read_csv(file_path, header=None)
		y = np.array(data[1])
		y[y == 'M'] = 1
		y[y == 'B'] = 0
		self.y = y.astype('int')
		self.X = np.array(data.iloc[:, 2:])
	
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
		print('----- {} score without any preprocessing: {} -----'.format(self.model_type, score))

	def preprocess_data_with_scaler(self):
		'''
		Preprocess data using sklearn StandardScaler to normalize the dataset.
		'''
		scaler = StandardScaler().fit(self.X_train)
		self.X_train_scaled = scaler.transform(self.X_train)
		self.X_test_scaled = scaler.transform(self.X_test)

	def score_after_preprocessing(self):
		self.preprocess_data_with_scaler()
		score = self.model_type.score(self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test)
		print('----- {} score after normalizing dataset: {} -----'.format(self.model_type, score))

	def train_and_predict_for_best_params(self, values, is_scaled=False):
		model = self.model_type.create_new_instance(values)
		if is_scaled:
			model.fit(self.X_train_scaled, self.y_train)
			model.predict(self.X_test_scaled)
			return model.score(self.X_test_scaled, self.y_test)
		else:
			model.fit(self.X_train, self.y_train)
			model.predict(self.X_test)
			return model.score(self.X_test, self.y_test)

	def grid_search_with_cross_validation(self, use_preprocessing=False, k_fold=CV_FOLD):
		'''
		Tries to find optimal value of paramters for a model by using cross validations and cv grid
		'''
		classifier = self.model_type.create_new_instance(values={})
		classifier_gscv = GridSearchCV(classifier, self.model_type.param_grid(), cv=k_fold)
		if use_preprocessing:
			classifier_gscv.fit(self.X_train_scaled, self.y_train)
			print('----- {} best param values using grid search cv for {}-fold cross validation on normalized dataset: {} -----'.format(self.model_type, k_fold, classifier_gscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_gscv.best_params_, is_scaled=True)
			print('----- {} score using grid search for {}-fold cross validation on normalized test dataset: {} -----'.format(self.model_type, k_fold, score))
		else:
			classifier_gscv.fit(self.X_train, self.y_train)
			print('----- {} best param values using grid search for {}-fold cross validation without any preprocessing: {} -----'.format(self.model_type, k_fold, classifier_gscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_gscv.best_params_)
			print('----- {} score using grid search for {}-fold cross validation on test dataset without any preprocessing: {} -----'.format(self.model_type, k_fold, score))

	def random_search_with_cross_validation(self, use_preprocessing=False, k_fold=CV_FOLD):
		'''
		Tries to find optimal value of paramters for a model by using cross validations and random search
		'''
		classifier = self.model_type.create_new_instance(values={})
		classifier_rscv = RandomizedSearchCV(classifier, self.model_type.param_grid(is_random=True), cv=k_fold)
		if use_preprocessing:
			classifier_rscv.fit(self.X_train_scaled, self.y_train)
			print('----- {} best param values using random search cv for {}-fold cross validation on normalized dataset: {} -----'.format(self.model_type, k_fold, classifier_rscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_rscv.best_params_, is_scaled=True)
			print('----- {} score using random search cv for {}-fold cross validation on normalized test dataset: {} -----'.format(self.model_type, k_fold, score))
		else:
			classifier_rscv.fit(self.X_train, self.y_train)
			print('----- {} best param values using random search cv for {}-fold cross validation without any preprocessing: {} -----'.format(self.model_type, k_fold, classifier_rscv.best_params_))
			score = self.train_and_predict_for_best_params(values=classifier_rscv.best_params_)
			print('----- {} score using random search cv for {}-fold cross validation on test dataset without any preprocessing: {} -----'.format(self.model_type, k_fold, score))

	def perform_experiments(self, file_path):
		self.get_data(file_path)
		self.get_score_without_any_processing()
		self.score_after_preprocessing()
		
		# skip grid and random search for GaussianNb as we don't have any hyper-params
		if self.model_type.__class__.__name__ != "GaussianNbClassifier":
			self.grid_search_with_cross_validation()
			self.grid_search_with_cross_validation(use_preprocessing=True)
			self.random_search_with_cross_validation()
			self.random_search_with_cross_validation(use_preprocessing=True)
