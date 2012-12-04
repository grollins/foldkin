import abc

class Judge(object):
    """docstring for Judge"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def judge_prediction(self, model, data_predictor, target_data):
        return


class SimpleJudge(Judge):
    """docstring for SimpleJudge"""
    def __init__(self):
        super(SimpleJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        prediction = data_predictor.predict_data(model)
        return prediction


class FoldRateJudge(Judge):
    """docstring for FoldRateJudge"""
    def __init__(self):
        super(FoldRateJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        return
