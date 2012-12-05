import abc
from sklearn.metrics import mean_squared_error

class Judge(object):
    """docstring for Judge"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def judge_prediction(self, model, data_predictor, target_data):
        return


class FoldRateJudge(Judge):
    """docstring for FoldRateJudge"""
    def __init__(self):
        super(FoldRateJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature = target_data.get_feature()
        target = target_data.get_target()
        prediction = data_predictor.predict_data(model, feature)
        return mean_squared_error(target, prediction), prediction
