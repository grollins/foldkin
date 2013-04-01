from foldkin.scipy_optimizer import ScipyOptimizer
import foldkin.one_param_curve.curve_fit_one_feature_model as curve
from foldkin.util import make_score_fcn
import nose.tools
from types import FloatType

EPSILON = 1e-3

@nose.tools.istest
class TestCurveFitOneFeatureFit(object):
    @nose.tools.istest
    def return_correct_score_and_optimal_parameter_value(self):
        '''This example fits a linear model to y=2x+5.
           The fitting function is y= a * x^b + c.
           The expected fit parameters are a = 2, b = 1, and c = 5.
        '''
        initial_parameters = curve.CurveFitOneFeatureParameterSet()
        initial_parameters.set_parameter('a', 0.0)
        initial_parameters.set_parameter('b', 1.0)
        initial_parameters.set_parameter('c', 0.0)
        judge = curve.CurveFitOneFeatureJudge()
        data_predictor = curve.CurveFitOneFeatureDataPredictor()
        target_data = curve.CurveFitOneFeatureTargetData()
        target_data.load_data()
        id_list = target_data.get_id_list()
        model_factory = curve.CurveFitOneFeatureModelFactory(id_list)
        score_fcn = make_score_fcn(model_factory, initial_parameters,
                                   judge, data_predictor, target_data)
        optimizer = ScipyOptimizer()
        results = optimizer.optimize_parameters(score_fcn, initial_parameters)
        new_params, score, num_iterations = results
        error_message = "Expected float, got %s %s" % (type(score), score)
        nose.tools.ok_(type(score) is FloatType, error_message)
        error_message = "Expected ParameterSet, got %s" % new_params
        nose.tools.ok_(type(new_params) is type(initial_parameters),
                       error_message)
        nose.tools.ok_(abs(0.0 - score) < EPSILON,
                       "Expected score = 2.0, got %s %s" % (type(score), score))
        nose.tools.ok_(abs(2.0 - new_params.get_parameter('a')) < EPSILON,
                       "Expected optimal a = 2.0, got %s" % new_params)
        nose.tools.ok_(abs(1.0 - new_params.get_parameter('b')) < EPSILON,
                       "Expected optimal b = 1.0, got %s" % new_params)
        nose.tools.ok_(abs(5.0 - new_params.get_parameter('c')) < EPSILON,
                       "Expected optimal c = 5.0, got %s" % new_params)
        return
