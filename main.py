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
	print("\n\n******** messidor_features data ***********\n")
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

	print("\n\n******** breast-cancer-wisconsin data ***********\n")
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

	print("\n\n******** statlog australian credit data ***********\n")
	print("*******************************************\n")
	
	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.perform_experiments('./data/statlog-australian-credit-data.data')

	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.perform_experiments('./data/statlog-australian-credit-data.data')

	gaussian_nb_classifier = GaussianNbClassifier()
	model = Model(model_type=gaussian_nb_classifier)
	model.perform_experiments('./data/statlog-australian-credit-data.data')

	nn_classifier = MlpClassifier()
	model = Model(model_type=nn_classifier)
	model.perform_experiments('./data/statlog-australian-credit-data.data')


	print("\n\n******** statlog german credit data ***********\n")
	print("*******************************************\n")

	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.perform_experiments('./data/statlog-german-credit-data.data')

	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.perform_experiments('./data/statlog-german-credit-data.data')

	gaussian_nb_classifier = GaussianNbClassifier()
	model = Model(model_type=gaussian_nb_classifier)
	model.perform_experiments('./data/statlog-german-credit-data.data')

	nn_classifier = MlpClassifier()
	model = Model(model_type=nn_classifier)
	model.perform_experiments('./data/statlog-german-credit-data.data')

	print("\n\n******** steel plates faults ***********\n")
	print("*******************************************\n")

	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.perform_experiments('./data/steel-plates-faults.NNA')

	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.perform_experiments('./data/steel-plates-faults.NNA')

	gaussian_nb_classifier = GaussianNbClassifier()
	model = Model(model_type=gaussian_nb_classifier)
	model.perform_experiments('./data/steel-plates-faults.NNA')

	nn_classifier = MlpClassifier()
	model = Model(model_type=nn_classifier)
	model.perform_experiments('./data/steel-plates-faults.NNA')

__main__()