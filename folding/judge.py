import abc

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
        return
