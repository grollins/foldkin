import numpy
from numpy import isnan
from base.judge import Judge
from sklearn.metrics import mean_squared_error
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.util import convert_T_to_beta

def print_nan_error_msg(target_array, prediction_array):
    error_msg = ""
    for t,p in zip(target_array, prediction_array):
        error_msg += "%.2e, %.2e\n" % (t, p)
    return error_msg

class FoldRateJudge(Judge):
    """docstring for FoldRateJudge"""
    def __init__(self):
        super(FoldRateJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data, noisy=False):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model)
        prediction_array = prediction.as_array()

        if noisy:
            for i,p in enumerate(prediction):
                print p, prediction_array[i], target_array[i]

        error_msg = "%s  %s" % (target_array.shape, prediction_array.shape)
        assert target_array.shape == prediction_array.shape, error_msg
        return mean_squared_error(target_array, prediction_array), prediction


class CoopCollectionJudge(Judge):
    def __init__(self):
        super(CoopCollectionJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data, noisy=False):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model)
        prediction_array = prediction.as_array_from_id_list(feature_array)

        if noisy:
            for i,f in enumerate(feature_array):
                print f, prediction_array[i], target_array[i]

        error_msg = "%s  %s" % (target_array.shape, prediction_array.shape)
        assert target_array.shape == prediction_array.shape, error_msg
        mse = mean_squared_error(target_array, prediction_array)
        return mse, prediction


class CoopCollectionR2Judge(Judge):
    def __init__(self):
        super(CoopCollectionR2Judge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data, noisy=False):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model)
        prediction_array = prediction.as_array_from_id_list(feature_array)

        if noisy:
            for i,f in enumerate(feature_array):
                print f, prediction_array[i], target_array[i]

        error_msg = "%s  %s" % (target_array.shape, prediction_array.shape)
        assert target_array.shape == prediction_array.shape, error_msg
        cc = numpy.corrcoef(target_array, prediction_array)[0,1]
        R2 = cc**2
        return R2, prediction


class TemperatureDependenceJudge(Judge):
    """docstring for TemperatureDependenceJudge"""
    def __init__(self):
        super(TemperatureDependenceJudge, self).__init__()
        self.mf = CoopModelFactory()

    def judge_prediction(self, fold_model, unfold_model, fold_data_predictor,
                         unfold_data_predictor, fold_target_data,
                         unfold_target_data, noisy=False):
        if fold_target_data:
            fold_feature_array = fold_target_data.get_feature()
            fold_target_array = fold_target_data.get_target()
            fold_prediction = fold_data_predictor.predict_data(fold_model)
            fold_prediction_array = fold_prediction.as_array()
            if noisy:
                for i,p in enumerate(fold_prediction):
                    print p, fold_prediction_array[i], fold_target_array[i]
            error_msg = "%s  %s" % (fold_target_array.shape, fold_prediction_array.shape)
            assert fold_target_array.shape == fold_prediction_array.shape, error_msg
            fold_mse = mean_squared_error(fold_target_array, fold_prediction_array)
            assert not isnan(fold_mse), print_nan_error_msg(fold_target_array,
                                                            fold_prediction_array)
        else:
            fold_mse = 0.0
            fold_prediction = None

        if unfold_target_data:
            unfold_feature_array = unfold_target_data.get_feature()
            unfold_target_array = unfold_target_data.get_target()
            unfold_prediction = unfold_data_predictor.predict_data(unfold_model)
            unfold_prediction_array = unfold_prediction.as_array()
            if noisy:
                for i,p in enumerate(unfold_prediction):
                    print p, unfold_prediction_array[i], unfold_target_array[i]
            error_msg = "%s  %s" % (unfold_target_array.shape, unfold_prediction_array.shape)
            assert unfold_target_array.shape == unfold_prediction_array.shape, error_msg
            unfold_mse = mean_squared_error(unfold_target_array, unfold_prediction_array)
            assert not isnan(unfold_mse), print_nan_error_msg(unfold_target_array,
                                                              unfold_prediction_array)
        else:
            unfold_mse = 0.0
            unfold_prediction = None
        score = (fold_mse + unfold_mse)/2.

        return score, fold_prediction, unfold_prediction
