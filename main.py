from models.model import Model
from models.knn_classifier import KnnClassifier
from models.svm_classifier import SvmClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)       # Ignore sklearn deprecation warnings

def __main__():
	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.get_data('./data/messidor_features.arff')
	model.get_score_without_any_processing()
	model.score_after_preprocessing()
	model.grid_search_with_cross_validation(k_fold=2)
	model.grid_search_with_cross_validation()
	model.grid_search_with_cross_validation(k_fold=2, use_preprocessing=True)
	model.grid_search_with_cross_validation(use_preprocessing=True)

	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.get_data('./data/messidor_features.arff')
	model.get_score_without_any_processing()
	model.score_after_preprocessing()
	model.grid_search_with_cross_validation(k_fold=2)
	model.grid_search_with_cross_validation()
	model.grid_search_with_cross_validation(k_fold=2, use_preprocessing=True)
	model.grid_search_with_cross_validation(use_preprocessing=True)

__main__()