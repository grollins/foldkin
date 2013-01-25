import numpy
from base.data_predictor import DataPredictor
from foldkin.fold_rate_prediction import FoldRatePrediction,\
                                         FoldRateCollectionPrediction

class FoldRatePredictor(DataPredictor):
    """docstring for FoldRatePredictor"""
    def __init__(self):
        super(FoldRatePredictor, self).__init__()
        self.prediction_factory = FoldRatePrediction

    def predict_data(self, model, feature=None):
        log_k1 = model.get_parameter('log_k1')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        Q = boltzmann_factor_array.sum()
        inds = range(len(boltzmann_factor_array))
        inds.remove(model.folded_index)
        Q_0 = boltzmann_factor_array[inds].sum()
        P1_eq = boltzmann_factor_array[model.first_excited_index]
        log_fold_rate = log_k1 + numpy.log10(P1_eq / Q_0)
        return self.prediction_factory(log_fold_rate)


class UnfoldRatePredictor(DataPredictor):
    """docstring for UnfoldRatePredictor"""
    def __init__(self):
        super(UnfoldRatePredictor, self).__init__()
        self.prediction_factory = FoldRatePrediction

    def predict_data(self, model, feature=None):
        log_k1 = model.get_parameter('log_k1')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        folded_weight = boltzmann_factor_array[model.folded_index]
        first_excited_weight = boltzmann_factor_array[model.first_excited_index]
        log_unfold_rate = log_k1 + numpy.log10(first_excited_weight / folded_weight)
        return self.prediction_factory(log_unfold_rate)


class FoldRateCollectionPredictor(DataPredictor):
    """docstring for FoldRateCollectionPredictor"""
    def __init__(self, element_predictor):
        super(FoldRateCollectionPredictor, self).__init__()
        self.element_predictor = element_predictor()
        self.prediction_factory = FoldRateCollectionPrediction

    def predict_data(self, model_collection):
        prediction_collection = self.prediction_factory()
        for this_element in model_collection:
            element_prediction = self.element_predictor.predict_data(this_element)
            prediction_collection.add_prediction(this_element.get_id(),
                                                 element_prediction)
        return prediction_collection

