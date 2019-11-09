from scipy.io import arff
import os
import random

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
from sklearn.preprocessing import LabelEncoder
import numpy as np

TEST_SIZE_SPLIT = 0.3
CV_FOLD = 7

class Model():

	def __init__(self, model_type):
		self.model_type = model_type

	def get_data(self, file_path):
		if file_path.endswith('arff'):
			if file_path.endswith('messidor_features.arff'):
				self.get_arff_data(file_path)
			elif file_path.endswith('seismic-bumps.arff'):
				self.get_seismic_bumps_arff_data(file_path)
			else:
				self.get_ThoraricSurgery_arff_data(file_path)
		elif file_path.endswith('data'):
			if file_path.endswith('adult.data'):
				self.get_adult_data(file_path)
			elif file_path.endswith('wisconsin.data'):
				self.get_bc_wisconsin_data(file_path)
			elif 'german' in file_path:
				self.get_german_credit_data(file_path)
			elif 'australian' in file_path:
				self.get_au_credit_data(file_path)
			else:
				self.get_yeast_data(file_path)
		elif file_path.endswith('NNA'):
			self.get_steel_plates_faults_data(file_path)
		elif file_path.endswith('xls'):
			self.get_default_of_credit_cards_clients_data(file_path)
		else:
			print("Don't know how to load this data")

	def get_adult_data(self, file_path):
		'''
		Replace <=50K as 0 and >50K as 1 to make classification work properly.
		Also, replace string features into numbers
		'''
		data = pd.read_csv(file_path, header=None)
		data = self.change_text_to_number_data_for_adult(data)
		data.replace(' <=50K', 0, inplace=True)
		data.replace(' >50K', 1, inplace=True)
		self.y = np.array(data.iloc[:,14])
		self.X = np.array(data.iloc[:,:14])


	def change_text_to_number_data_for_adult(self, data):
		'''
		Replace textual data in columns with numbers
		'''
		random.seed(0)

		workclass_dict = {' State-gov': 0, ' Self-emp-not-inc': 1, ' Private': 2, ' Federal-gov': 3, ' Local-gov': 4, ' Self-emp-inc': 5, ' Without-pay': 6, ' Never-worked': 7, ' ?': random.randrange(0,8)}
		education_dict = {' Bachelors': 0, ' HS-grad': 1, ' 11th': 2, ' Masters': 3, ' 9th': 4, ' Some-college': 5, ' Assoc-acdm': 6, ' Assoc-voc': 7, ' 7th-8th': 8, ' Doctorate': 9, ' Prof-school': 10, ' 5th-6th': 11, ' 10th': 12, ' 1st-4th': 13, ' Preschool': 14, ' 12th': 15, ' ?': random.randrange(0,16)}
		marital_status_dict = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 2, ' Married-spouse-absent': 3, ' Separated': 4, ' Married-AF-spouse': 5, ' Widowed': 6, ' ?': random.randrange(0,7)}
		occupation_dict = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2, ' Prof-specialty': 3, ' Other-service': 4, ' Sales': 5, ' Craft-repair': 6, ' Transport-moving': 7, ' Farming-fishing': 8, ' Machine-op-inspct': 9, ' Tech-support': 10, ' Protective-serv': 11, ' Armed-Forces': 12, ' Priv-house-serv': 13, ' ?': random.randrange(0,14)}
		relationship_dict = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, ' Own-child': 3, ' Unmarried': 4, ' Other-relative': 5, ' ?': random.randrange(0,6)}
		race_dict = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4, ' ?': random.randrange(0,5)}
		sex_dict = {' Male': 0, ' Female': 1, ' ?': random.randrange(0,2)}
		country_dict = {' United-States': 0, ' Cuba': 1, ' Jamaica': 2, ' India': 3, ' Mexico': 4, ' South': 5, ' Puerto-Rico': 6, ' Honduras': 7, ' England': 8, ' Canada': 9, ' Germany': 10, ' Iran': 11, ' Philippines': 12, ' Italy': 13, ' Poland': 14, ' Columbia': 15, ' Cambodia': 16, ' Thailand': 17, ' Ecuador': 18, ' Laos': 19, ' Taiwan': 20, ' Haiti': 21, ' Portugal': 22, ' Dominican-Republic': 23, ' El-Salvador': 24, ' France': 25, ' Guatemala': 26, ' China': 27, ' Japan': 28, ' Yugoslavia': 29, ' Peru': 30, ' Outlying-US(Guam-USVI-etc)': 31, ' Scotland': 32, ' Trinadad&Tobago': 33, ' Greece': 34, ' Nicaragua': 35, ' Vietnam': 36, ' Hong': 37, ' Ireland': 38, ' Hungary': 39, ' Holand-Netherlands': 40, ' ?': random.randrange(0,41)}
		
		for key, val in workclass_dict.items():
			data[1].replace(key, val, inplace=True)

		for key, val in education_dict.items():
			data[3].replace(key, val, inplace=True)

		for key, val in marital_status_dict.items():
			data[5].replace(key, val, inplace=True)

		for key, val in occupation_dict.items():
			data[6].replace(key, val, inplace=True)

		for key, val in relationship_dict.items():
			data[7].replace(key, val, inplace=True)

		for key, val in race_dict.items():
			data[8].replace(key, val, inplace=True)

		for key, val in sex_dict.items():
			data[9].replace(key, val, inplace=True)

		for key, val in country_dict.items():
			data[13].replace(key, val, inplace=True)
		
		return data

	def process_and_load_adult_test_data(self):
		data = pd.read_csv('./data/adult.test', header=None)
		data = self.change_text_to_number_data_for_adult(data)
		data.replace(' <=50K.', 0, inplace=True)
		data.replace(' >50K.', 1, inplace=True)
		self.X_train = self.X
		self.y_train = self.y
		self.y_test = np.array(data.iloc[:,14])
		self.X_test = np.array(data.iloc[:,:14])

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

	def get_yeast_data(self, file_path):
		'''
        This method is used to load the data from a file which has .data extension and seperate out X and y labels
        '''
		names = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
		data = pd.read_csv(file_path, names=names, delim_whitespace=True)
		dataset = pd.DataFrame(data)
		label_encoder = LabelEncoder()
		dataset['Sequence Name'] = label_encoder.fit_transform(dataset['Sequence Name'])
		dataset['class'] = label_encoder.fit_transform(dataset['class'])
		self.X = dataset.iloc[:, 0:9].values.astype(int)
		self.y = dataset.iloc[:, 9].values

	def get_seismic_bumps_arff_data(self, file_path):
		'''
        This method is used to load the arff data and seperate out X and y labels
        '''
		data = arff.loadarff(file_path)
		dataset = pd.DataFrame(data[0])
		label_encoder = LabelEncoder()
		dataset['seismic'] = label_encoder.fit_transform(dataset['seismic'])
		dataset['seismoacoustic'] = label_encoder.fit_transform(dataset['seismoacoustic'])
		dataset['shift'] = label_encoder.fit_transform(dataset['shift'])
		dataset['ghazard'] = label_encoder.fit_transform(dataset['ghazard'])
		dataset['class'] = label_encoder.fit_transform(dataset['class'])
		self.X = dataset.iloc[:, 0: 18].values.astype(int)
		self.y = dataset.iloc[:, 18].values

	def get_ThoraricSurgery_arff_data(self, file_path):
		'''
        This method is used to load the arff data and seperate out X and y labels
        '''
		data = arff.loadarff(file_path)
		dataset = pd.DataFrame(data[0])
		label_encoder = LabelEncoder()
		dataset['DGN'] = label_encoder.fit_transform(dataset['DGN'])
		dataset['PRE6'] = label_encoder.fit_transform(dataset['PRE6'])
		dataset['PRE7'] = label_encoder.fit_transform(dataset['PRE7'])
		dataset['PRE8'] = label_encoder.fit_transform(dataset['PRE8'])
		dataset['PRE9'] = label_encoder.fit_transform(dataset['PRE9'])
		dataset['PRE10'] = label_encoder.fit_transform(dataset['PRE10'])
		dataset['PRE11'] = label_encoder.fit_transform(dataset['PRE11'])
		dataset['PRE14'] = label_encoder.fit_transform(dataset['PRE14'])
		dataset['PRE17'] = label_encoder.fit_transform(dataset['PRE17'])
		dataset['PRE19'] = label_encoder.fit_transform(dataset['PRE19'])
		dataset['PRE25'] = label_encoder.fit_transform(dataset['PRE25'])
		dataset['PRE30'] = label_encoder.fit_transform(dataset['PRE30'])
		dataset['PRE32'] = label_encoder.fit_transform(dataset['PRE32'])
		dataset['Risk1Yr'] = label_encoder.fit_transform(dataset['Risk1Yr'])
		self.X = dataset.iloc[:, 0: 16].values
		self.y = dataset.iloc[:, 16].values

	def get_default_of_credit_cards_clients_data(self, file_path):
		'''
        This method is used to load the xls data and separate out X and y labels
        '''
		data = pd.read_excel(file_path)
		dataset = pd.DataFrame(data)
		dataset = dataset.iloc[1:, 1:]
		self.X = dataset.iloc[:, 0:23].values.astype(int)
		self.y = dataset.iloc[:, 23].values.astype(int)

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
		if file_path != "./data/adult.data":
			self.get_train_and_test_split()
		else:
			self.process_and_load_adult_test_data()

		self.get_score_without_any_processing()
		self.score_after_preprocessing()
		
		# skip grid and random search for GaussianNb as we don't have any hyper-params
		if self.model_type.__class__.__name__ != "GaussianNbClassifier":
			self.grid_search_with_cross_validation()
			self.grid_search_with_cross_validation(use_preprocessing=True)
			self.random_search_with_cross_validation()
			self.random_search_with_cross_validation(use_preprocessing=True)
