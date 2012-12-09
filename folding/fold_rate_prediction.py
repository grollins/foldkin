import numpy
import base.prediction

class SingleFoldRatePrediction(base.prediction.Prediction):
    """docstring for SingleFoldRatePrediction"""
    def __init__(self, log_fold_rate):
        super(SingleFoldRatePrediction, self).__init__()
        self.log_fold_rate = log_fold_rate

    def as_array(self):
        return numpy.array([self.log_fold_rate])

    def compute_difference(self, other_log_fold_rate):
        return self.log_fold_rate - other_log_fold_rate


class FoldRateCollectionPrediction(base.prediction.Prediction):
    """docstring for FoldRateCollectionPrediction"""
    def __init__(self):
        super(FoldRateCollectionPrediction, self).__init__()
        self.log_fold_rate_list = []

    def __iter__(self):
        for log_fold_rate in self.log_fold_rate_list:
            yield log_fold_rate

    def as_array(self):
        return numpy.array(self.log_fold_rate_list)

    def add_prediction(self, prediction):
        prediction_array = prediction.as_array()
        assert len(prediction_array) == 1
        self.log_fold_rate_list.append( prediction_array[0] )
