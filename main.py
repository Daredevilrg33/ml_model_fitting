from models.model import Model
from models.knn_classifier import KnnClassifier
from models.svm_classifier import SvmClassifier

def __main__():
	knn_classifier = KnnClassifier()
	model = Model(model_type=knn_classifier)
	model.get_data('./data/messidor_features.arff')
	model.get_score_without_any_processing()
	model.get_kfold_cross_validation()
	model.grid_search_with_cross_validation()

	svm_classifier = SvmClassifier()
	model = Model(model_type=svm_classifier)
	model.get_data('./data/messidor_features.arff')
	model.get_score_without_any_processing()
	model.get_kfold_cross_validation()

__main__()