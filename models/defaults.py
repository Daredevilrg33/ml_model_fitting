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
}