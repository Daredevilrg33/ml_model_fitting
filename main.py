from models.model import Model
from models.knn_classifier import KnnClassifier
from models.svm_classifier import SvmClassifier
from models.gaussian_nb_classifier import GaussianNbClassifier
from models.mlp_classifier import MlpClassifier
from models.logistic_reg_classifier import LogisticRegClassifier

import warnings

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)       # Ignore sklearn deprecation warnings

@ignore_warnings(category=ConvergenceWarning)
def __main__():
	
	dataset_list = [
		'./data/messidor_features.arff',
		'./data/breast-cancer-wisconsin.data',
		'./data/statlog-australian-credit-data.data',
		'./data/statlog-german-credit-data.data',
		'./data/steel-plates-faults.NNA',
		'./data/adult.data',
	]


	for dataset in dataset_list:
		print("\n\n******** {} data ***********\n".format(dataset.split('/')[-1]))
		print("*******************************************\n")

		knn_classifier = KnnClassifier(dataset)
		model = Model(model_type=knn_classifier)
		model.perform_experiments(dataset)
		
		svm_classifier = SvmClassifier(dataset)
		model = Model(model_type=svm_classifier)
		model.perform_experiments(dataset)

		gaussian_nb_classifier = GaussianNbClassifier(dataset)
		model = Model(model_type=gaussian_nb_classifier)
		model.perform_experiments(dataset)

		nn_classifier = MlpClassifier(dataset)
		model = Model(model_type=nn_classifier)
		model.perform_experiments(dataset)

		lr_classifier = LogisticRegClassifier(dataset)
		model = Model(model_type=lr_classifier)
		model.perform_experiments(dataset)


__main__()