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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
				'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.01, 0.10, 1.0, 10]
			}
		},
		'gaussian_nb': {
			'defaults': {

			},
			'param_grid': {
				'var_smoothing': np.arange(0.1, 1, .1)
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

			}
		},
		'dt': {
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
			}
		}
	},
	'./data_regression/dataset_Facebook.csv': {
		'linear_reg': {
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
			}
		}
	},
	'./data_regression/sgemm_product.csv': {
		'linear_reg': {
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
			}
		}
	},
	'./data_regression/winequality-red.csv': {
		'linear_reg': {
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
			}
		}
	},
	'./data_regression/communities.data': {
		'linear_reg': {
			'defaults': {
			},
			'param_grid': {
			}
		}
	}
}