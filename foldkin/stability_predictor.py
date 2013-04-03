import numpy
from base.data_predictor import DataPredictor
from base.prediction import Prediction
from foldkin.util import ALMOST_ZERO, ALMOST_INF

class StabilityPrediction(Prediction):
    """docstring for StabilityPrediction"""
    def __init__(self, stability):
        super(StabilityPrediction, self).__init__()
        self.stability = stability

    def __str__(self):
        return "%.3f" % self.stability

    def __iter__(self):
        yield self.stability

    def as_array(self):
        return numpy.array([self.stability])

    def compute_difference(self, other_stability):
        return self.stability - other_stability


class StabilityPredictor(DataPredictor):
    """docstring for StabilityPredictor"""
    def __init__(self):
        super(StabilityPredictor, self).__init__()
        self.prediction_factory = StabilityPrediction

    def predict_data(self, model):
        return self.predict_stability(model)

    def predict_stability(self, model):
        ps = model.get_parameter_set()
        beta = ps.get_parameter('beta')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        boltzmann_factor_array[numpy.isinf(boltzmann_factor_array)] = ALMOST_INF
        boltzmann_factor_array[numpy.isnan(boltzmann_factor_array)] = ALMOST_ZERO
        Q = boltzmann_factor_array.sum()
        inds = range(len(boltzmann_factor_array))
        inds.remove(model.folded_index)
        Q_u = boltzmann_factor_array[inds].sum()
        Q_f = boltzmann_factor_array[model.folded_index]
        Gf = -numpy.log(Q_f / Q) / beta
        Gu = -numpy.log(Q_u / Q) / beta
        dG = Gf - Gu
        return self.prediction_factory(dG)
