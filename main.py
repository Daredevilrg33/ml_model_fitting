from models.model import Model
from models.knn_classifier import KnnClassifier
from models.svm_classifier import SvmClassifier
from models.gaussian_nb_classifier import GaussianNbClassifier
from models.mlp_classifier import MlpClassifier
import warnings

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)       # Ignore sklearn deprecation warnings

@ignore_warnings(category=ConvergenceWarning)
def __main__():
	print("******** messidor_features data ***********\n")
	print("*******************************************\n")

	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.perform_experiments('./data/messidor_features.arff')
	
	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.perform_experiments('./data/messidor_features.arff')

	gaussian_nb_classifier = GaussianNbClassifier()
	model = Model(model_type=gaussian_nb_classifier)
	model.perform_experiments('./data/messidor_features.arff')

	nn_classifier = MlpClassifier()
	model = Model(model_type=nn_classifier)
	model.perform_experiments('./data/messidor_features.arff')

	print("******** breast-cancer-wisconsin data ***********\n")
	print("*******************************************\n")
	
	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.perform_experiments('./data/breast-cancer-wisconsin.data')

	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.perform_experiments('./data/breast-cancer-wisconsin.data')

	gaussian_nb_classifier = GaussianNbClassifier()
	model = Model(model_type=gaussian_nb_classifier)
	model.perform_experiments('./data/breast-cancer-wisconsin.data')

	nn_classifier = MlpClassifier()
	model = Model(model_type=nn_classifier)
	model.perform_experiments('./data/breast-cancer-wisconsin.data')

__main__()