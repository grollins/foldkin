import numpy
from base.judge import Judge
from sklearn.metrics import mean_squared_error
from foldkin.util import make_copy

class CoopFitCurveJudge(Judge):
    """docstring for CoopFitCurveJudge"""
    def __init__(self):
        super(CoopFitCurveJudge, self).__init__()

    def judge_prediction(self, model, model_factory, fold_rate_predictor,
                         unfold_rate_predictor, quad_fit_to_kf_params,
                         quad_fit_to_ku_params, noisy=False):
        if fold_rate_predictor:
            x = quad_fit_to_kf_params.get_parameter('x')
            y0 = quad_fit_to_kf_params.get_parameter('y0')
            y2 = quad_fit_to_kf_params.get_parameter('y2')
            beta = model.get_parameter('beta')
            assert x == beta, '%.2f %.2f' % (x, beta)
            fold_prediction = fold_rate_predictor.predict_data(model)
            log_kf = fold_prediction.as_array()[0]
            parameter_set = make_copy(model.get_parameter_set())
            dlogkf_dbeta = fold_rate_predictor.predict_deriv(beta, parameter_set,
                                                             model_factory)
            parameter_set = make_copy(model.get_parameter_set())
            ddlogkf_dbeta = fold_rate_predictor.predict_second_deriv(beta,
                                                             parameter_set,
                                                             model_factory)
            fold_score1 = ((log_kf - y0)**2) / y0**2
            fold_score2 = (dlogkf_dbeta - 0)**2 / 1.
            fold_score3 = ((0.5 * ddlogkf_dbeta - y2)**2) / y2**2
            fold_score = numpy.sqrt((fold_score1 + fold_score2 + fold_score3)/3.)
            # fold_score = numpy.sqrt((fold_score1 + fold_score2)/2.)
            error_msg = "%.2f %.2f %.2f" % (fold_score1, fold_score2, fold_score3)
            assert not numpy.isnan(fold_score), error_msg
        else:
            fold_score = 0.0

        if unfold_rate_predictor:
            x = quad_fit_to_ku_params.get_parameter('x')
            y0 = quad_fit_to_ku_params.get_parameter('y0')
            y1 = quad_fit_to_ku_params.get_parameter('y1')
            beta = model.get_parameter('beta')
            assert x == beta, '%.2f %.2f' % (x, beta)
            unfold_prediction = unfold_rate_predictor.predict_data(model)
            log_ku = unfold_prediction.as_array()[0]
            dlogku_dbeta = unfold_rate_predictor.predict_deriv(model)
            unfold_score1 = (log_kf - y0)**2
            unfold_score2 = (dlogku_dbeta - y1)**2
            unfold_score = numpy.sqrt((unfold_score1 + unfold_score2)/2.)
        else:
            unfold_score = 0.0

        return (fold_score + unfold_score)/2.
