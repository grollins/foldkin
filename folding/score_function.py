import abc
from sklearn.metrics import mean_squared_error

class ScoreFunction(object):
    """ScoreFunction is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_score(self, true_data, predicted_data):
        return


class FoldingRateScoreFunction(ScoreFunction):
    """docstring for FoldingRateScoreFunction"""
    def __init__(self):
        super(FoldingRateScoreFunction, self).__init__()

    def compute_score(self, true_data_set, predicted_data_set):
        true_array = true_data_set.as_array()
        predicted_array = predicted_data_set.as_array()
        return mean_squared_error(true_array, predicted_array)


class SimpleScoreFunction(ScoreFunction):
    def compute_score(self, true_data, predicted_data):
        # in this example, the data is the score
        return predicted_data.y
