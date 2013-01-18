import numpy
from foldkin.base.prediction import Prediction

class SingleContactOrderPrediction(Prediction):
    """docstring for SingleContactOrderPrediction"""
    def __init__(self, log_fold_rate):
        super(SingleContactOrderPrediction, self).__init__()
        self.log_fold_rate = log_fold_rate

    def as_array(self):
        return numpy.array([self.log_fold_rate])

    def compute_difference(self, other_log_fold_rate):
        return self.log_fold_rate - other_log_fold_rate


class ContactOrderCollectionPrediction(Prediction):
    """docstring for ContactOrderCollectionPrediction"""
    def __init__(self):
        super(ContactOrderCollectionPrediction, self).__init__()
        self.pdb_id_list = []
        self.log_fold_rate_list = []

    def __iter__(self):
        for pdb_id, log_fold_rate in zip(self.pdb_id_list, self.log_fold_rate_list):
            yield pdb_id, log_fold_rate

    def as_array(self):
        return numpy.array(self.log_fold_rate_list)

    def add_prediction(self, pdb_id, prediction):
        self.pdb_id_list.append(pdb_id)
        prediction_array = prediction.as_array()
        assert len(prediction_array) == 1
        self.log_fold_rate_list.append( prediction_array[0] )
