import numpy
from base.data_predictor import DataPredictor
from foldkin.fold_rate_prediction import FoldRatePrediction,\
                                         FoldRateCollectionPrediction
from foldkin.util import ALMOST_ZERO, ALMOST_INF

class FoldRatePredictor(DataPredictor):
    """docstring for FoldRatePredictor"""
    def __init__(self):
        super(FoldRatePredictor, self).__init__()
        self.prediction_factory = FoldRatePrediction

    def _generate_nan_error_msg(self, model, bf_array):
        error_msg = ""
        for i, bf in enumerate(bf_array):
            error_msg += "%d  %.2e\n" % (i, bf)
        Q = bf_array.sum()
        inds = range(len(bf_array))
        inds.remove(model.folded_index)
        unfolded_weight = bf_array[inds].sum()
        first_excited_weight = bf_array[model.first_excited_index]
        error_msg += "unfolded: %.2e\n" % unfolded_weight
        error_msg += "1st excited state: %.2e\n" % first_excited_weight
        return error_msg

    def predict_data(self, model, feature=None):
        log_k1 = model.get_parameter('log_k1')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        boltzmann_factor_array[numpy.isinf(boltzmann_factor_array)] = ALMOST_INF
        boltzmann_factor_array[numpy.isnan(boltzmann_factor_array)] = ALMOST_ZERO
        Q = boltzmann_factor_array.sum()
        inds = range(len(boltzmann_factor_array))
        inds.remove(model.folded_index)
        Q_0 = boltzmann_factor_array[inds].sum()
        P1_eq = boltzmann_factor_array[model.first_excited_index]
        if P1_eq < ALMOST_ZERO:
            P1_eq = ALMOST_ZERO
        log_fold_rate = log_k1 + numpy.log10(P1_eq / Q_0)
        assert not numpy.isnan(log_fold_rate), self._generate_nan_error_msg(model, boltzmann_factor_array)
        return self.prediction_factory(log_fold_rate)


class UnfoldRatePredictor(DataPredictor):
    """docstring for UnfoldRatePredictor"""
    def __init__(self):
        super(UnfoldRatePredictor, self).__init__()
        self.prediction_factory = FoldRatePrediction

    def _generate_nan_error_msg(self, model, bf_array):
        error_msg = ""
        for i, bf in enumerate(bf_array):
            error_msg += "%d  %.2e\n" % (i, bf)
        folded_weight = bf_array[model.folded_index]
        first_excited_weight = bf_array[model.first_excited_index]
        error_msg += "folded: %.2e\n" % folded_weight
        error_msg += "1st excited state: %.2e\n" % first_excited_weight
        return error_msg

    def predict_data(self, model, feature=None):
        log_k1 = model.get_parameter('log_k1')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        boltzmann_factor_array[numpy.isinf(boltzmann_factor_array)] = ALMOST_INF
        boltzmann_factor_array[numpy.isnan(boltzmann_factor_array)] = ALMOST_ZERO
        folded_weight = boltzmann_factor_array[model.folded_index]
        first_excited_weight = boltzmann_factor_array[model.first_excited_index]
        if first_excited_weight < ALMOST_ZERO:
            first_excited_weight = ALMOST_ZERO
        if folded_weight < ALMOST_ZERO:
            folded_weight = ALMOST_ZERO
        log_unfold_rate = log_k1 + numpy.log10(first_excited_weight / folded_weight)
        assert not numpy.isnan(log_unfold_rate), self._generate_nan_error_msg(model, boltzmann_factor_array)
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

