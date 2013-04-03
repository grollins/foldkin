from base.judge import Judge
from sklearn.metrics import mean_squared_error

class StabilityJudge(Judge):
    """docstring for StabilityJudge"""
    def __init__(self):
        super(StabilityJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data, noisy=False):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model)
        prediction_array = prediction.as_array()

        error_msg = "%s  %s" % (target_array.shape, prediction_array.shape)
        assert target_array.shape == prediction_array.shape, error_msg
        mse = mean_squared_error(target_array, prediction_array)
        if noisy:
            print prediction_array, target_array, mse
        return mse, prediction
