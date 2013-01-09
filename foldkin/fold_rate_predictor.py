import numpy
import base.data_predictor
from foldkin.fold_rate_prediction import SingleFoldRatePrediction,\
                                         FoldRateCollectionPrediction

class SingleFoldRatePredictor(base.data_predictor.DataPredictor):
    """docstring for FoldRatePredictor"""
    def __init__(self):
        super(SingleFoldRatePredictor, self).__init__()
        self.prediction_factory = SingleFoldRatePrediction

    def predict_data(self, model, feature):
        log_k1 = model.get_parameter('log_k1')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        Q = boltzmann_factor_array.sum()
        inds = range(len(boltzmann_factor_array))
        inds.remove(model.folded_index)
        Q_0 = boltzmann_factor_array[inds].sum()
        P1_eq = boltzmann_factor_array[model.first_excited_index]
        log_fold_rate = log_k1 + numpy.log10(P1_eq / Q_0)
        return self.prediction_factory(log_fold_rate)


class FoldRateCollectionPredictor(base.data_predictor.DataPredictor):
    """docstring for FoldRateCollectionPredictor"""
    def __init__(self):
        super(FoldRateCollectionPredictor, self).__init__()
        self.single_rate_predictor = SingleFoldRatePredictor()
        self.prediction_factory = FoldRateCollectionPrediction

    def predict_data(self, model, feature):
        prediction_collection = self.prediction_factory()
        for this_value in feature:
            fold_rate_model = model.get_element(this_value)
            if fold_rate_model:
                element_prediction = self.single_rate_predictor.predict_data(fold_rate_model,
                                                        numpy.array(this_value))
                prediction_collection.add_prediction(element_prediction)
        return prediction_collection
