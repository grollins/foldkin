import numpy
from base.prediction import Prediction

class FoldRatePrediction(Prediction):
    """docstring for FoldRatePrediction"""
    def __init__(self, log_fold_rate):
        super(FoldRatePrediction, self).__init__()
        self.log_fold_rate = log_fold_rate

    def as_array(self):
        return numpy.array([self.log_fold_rate])

    def compute_difference(self, other_log_fold_rate):
        return self.log_fold_rate - other_log_fold_rate


class FoldRateCollectionPrediction(Prediction):
    """docstring for FoldRateCollectionPrediction"""
    def __init__(self):
        super(FoldRateCollectionPrediction, self).__init__()
        self.id_list = []
        self.log_fold_rate_list = []
        self.log_fold_rate_dict = {}

    def __iter__(self):
        for id_str, log_fold_rate in zip(self.id_list, self.log_fold_rate_list):
            yield id_str, log_fold_rate

    def as_array(self):
        return numpy.array(self.log_fold_rate_list)

    def as_array_from_id_list(self, id_list):
        rate_list = []
        for this_id in id_list:
            this_rate = self.log_fold_rate_dict[this_id]
            rate_list.append(this_rate)
        return numpy.array(rate_list)

    def add_prediction(self, id_str, prediction):
        self.id_list.append(id_str)
        prediction_array = prediction.as_array()
        assert len(prediction_array) == 1
        self.log_fold_rate_list.append( prediction_array[0] )
        self.log_fold_rate_dict[id_str] = prediction_array[0]
