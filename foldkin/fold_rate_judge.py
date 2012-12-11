import base.judge
from sklearn.metrics import mean_squared_error

class FoldRateJudge(base.judge.Judge):
    """docstring for FoldRateJudge"""
    def __init__(self):
        super(FoldRateJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model, feature_array)
        prediction_array = prediction.as_array()
        assert target_array.shape == prediction_array.shape
        return mean_squared_error(target_array, prediction_array), prediction
