import abc
import numpy

class Prediction(object):
    """Prediction is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def as_array(self):
        return


class SingleFoldRatePrediction(Prediction):
    """docstring for SingleFoldRatePrediction"""
    def __init__(self, log_fold_rate):
        super(SingleFoldRatePrediction, self).__init__()
        self.log_fold_rate = log_fold_rate

    def as_array(self):
        return numpy.array([self.log_fold_rate])

    def compute_difference(self, other_log_fold_rate):
        return self.log_fold_rate - other_log_fold_rate

