import numpy as np
import random

DEFAULTS = {
	'./data/messidor_features.arff': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,1]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {


			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],
			}
		},
		'dt': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [20,30,40,50],
			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/breast-cancer-wisconsin.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],
			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [3,5,10,15,20],
				"min_samples_leaf": [2,3,5],
				"max_leaf_nodes": [20,30,40,50],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/statlog-australian-credit-data.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],

			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [3,2,4,5],
				"min_samples_leaf": [5,10,15,20,25],
				"max_leaf_nodes": [20,30,40],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/statlog-german-credit-data.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],
			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/steel-plates-faults.NNA': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [3,5,6,7],
				"max_depth": [5,10,15],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/adult.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18), 
				'max_iter': 500, 
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))], 
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/seismic-bumps.arff': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/yeast.data': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/default_of_credit_card_clients.xls': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data/ThoraricSurgery.arff': {
		'knn': {
			'defaults': {
				'n_neighbors': 10
			},
			'param_grid': {
				'n_neighbors': np.arange(1, 30)
			}
		},
		'svm': {
			'defaults': {
				'kernel': 'rbf',
				'C': 1.0,
				'gamma': 'auto',
				'random_state': 0,
			},
			'param_grid': {
				'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': [0.00001,0.001,0.1,10]
			}
		},
		'nn':{
			'defaults': {
				'hidden_layer_sizes': (20, 18),
				'max_iter': 500,
				'random_state': 0,
			},
			'param_grid': {
				'hidden_layer_sizes': [(random.randrange(1, 20), random.randrange(1, 20)), (random.randrange(1, 20), random.randrange(1, 20))],
				'max_iter': [random.randrange(100, 700)], 'activation':['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'alpha':[0.001, 0.0001, 0.00001],
			}
		},
		'logistic_reg': {
			'defaults': {

			},
			'param_grid': {
				'penalty': ['l1', 'l2'],
				'C': [0.01, 0.1, 1, 10, 100],
				'class_weight': [{1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
				'solver': ['liblinear', 'saga'],


			}
		},
		'dt': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/bike_sharing_hour.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept':[True, False],
				'normalize':[True, False],
				'copy_X':[True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
              	"min_samples_split": [10, 15, 20],
              	"max_depth": [2, 6, 8],
              	"min_samples_leaf": [20, 40, 100],
              	"max_leaf_nodes": [50,100],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [0.1,1, 10],
			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/dataset_Facebook.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
'random_state': [0],
				"min_samples_split": [10, 15, 20, 40],
				"max_depth": [4,6,8, 10],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [40,50, 100],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/qsar_aquatic_toxicity.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
				"min_samples_split": [10, 15, 20, 40],
				"max_depth": [4,6,8, 10],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [40,50, 100],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/sgemm_product.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0]
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/Concrete_Data.xls': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
  				'random_state': [0],
				"min_samples_split": [2,3,5],
				"max_depth": [5,10,15,20],
				"min_samples_leaf": [5,10,15],
				"max_leaf_nodes": [40,50, 100, 120],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/winequality-red.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
				'max_depth':[None, 5, 3, 1],
                'min_samples_split':np.linspace(0.1, 1.0, 5, endpoint=True),
                'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
                'max_features':['auto', 'sqrt', 'log2'],
			}
		},
		'svr': {
			'defaults': {
				'C':100,
				'gamma':0.1,
				'epsilon':.1
			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/winequality-white.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
				'random_state': [0],
				'max_depth':[None, 5, 3, 1],
                'min_samples_split':np.linspace(0.1, 1.0, 5, endpoint=True),
                'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
                'max_features':['auto', 'sqrt', 'log2'],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./cfar10/data_batch': {
		'dt': {
			'defaults': {
				'max_depth': 10000, 'random_state': 0, 'min_samples_split': 5,
			},
			'param_grid': {
				'max_depth': [100, 500, 1000, 10000], 'max_features': ['auto', 'log2', None], 'criterion': ['gini', 'entropy'],
			}
		},
	},
  	'./data_regression/student-por.csv': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
                'max_depth':np.linspace(1, 32, 16, endpoint=True),
                'min_samples_split':np.linspace(0.1, 1.0, 5, endpoint=True),
                'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True),
                # 'max_features':list(range(1,train.shape[1])),
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
			'param_grid': {
                'max_depth': [ 40,  70,  100],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [ 1800, 2000]}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/communities.data': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
				"min_samples_split": [5,10],
				"max_depth": [8, 10,14],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [10,20,40],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/ACT2_competition_training.npz': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
				"min_samples_split": [4,5,6],
				"max_depth": [8, 10,14],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [40,50],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [0.1, 1, 10],
			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/ACT4_competition_training.npz': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
                'random_state': [0],
				"min_samples_split": [3,4,5],
				"max_depth": [ 10,14,16],
				"min_samples_leaf": [10,20, 40],
				"max_leaf_nodes": [50,60],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [0.1,1, 10],
                'kernel': ['rbf','sigmoid'],
                'gamma': ['auto','scale']

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	},
	'./data_regression/parkinson_train_data.txt': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
				'fit_intercept': [True, False],
				'normalize': [True, False],
				'copy_X': [True, False],
			}
		},
		'dtr': {
			'defaults': {
				'random_state': 0
			},
			'param_grid': {
 'random_state': [0],
				"min_samples_split": [2,3,4],
				"max_depth": [3,4,5],
				"min_samples_leaf": [40,60,80],
				"max_leaf_nodes": [10,20,30],
			}
		},
		'svr': {
			'defaults': {

			},
			'param_grid': {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf','sigmoid'],
                'gamma': [0.01, 0.10, 1.0, 10]

			}
		},
		'gaussian_pr': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'rf': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'nn': {
			'defaults': {

			},
			'param_grid': {

			}
		},
		'ab': {
			'defaults': {

			},
			'param_grid': {

			}
		}
	}
}