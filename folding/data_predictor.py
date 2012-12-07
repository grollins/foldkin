import abc
import numpy

class DataPredictor(object):
    """DataPredictor is an abstract class"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_data(self, model, feature):
        return


class FoldRatePredictor(object):
    """docstring for FoldRatePredictor"""
    def __init__(self):
        super(FoldRatePredictor, self).__init__()

    def predict_data(self, model, feature):
        log_k1 = model.get_parameter('log_k1')
        boltzmann_factor_array = model.compute_boltzmann_factors()
        Q = boltzmann_factor_array.sum()
        inds = range(len(boltzmann_factor_array))
        inds.remove(model.folded_index)
        Q_0 = boltzmann_factor_array[inds].sum()
        P1_eq = boltzmann_factor_array[model.first_excited_index]
        log_kf = log_k1 + numpy.log10(P1_eq / Q_0)
        return numpy.array( [log_kf] )
