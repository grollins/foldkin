from folding.scipy_optimizer import ScipyOptimizer
import folding.linear_model as lm
import numpy
import nose.tools
import types

EPSILON = 1e-3

@nose.tools.istest
class TestLinearFit(object):
    @nose.tools.istest
    def return_correct_score_and_optimal_parameter_value(self):
        '''This example fits a linear model to y=2x+5.
        '''
        model_factory = lm.LinearModelFactory()
        initial_parameters = lm.LinearParameterSet()
        initial_parameters.set_parameter('a', 0.0)
        initial_parameters.set_parameter('b', 0.0)
        judge = lm.LinearJudge()
        data_predictor = lm.LinearDataPredictor()
        target_data = lm.LinearTargetData()
        target_data.load_data()
        optimizer = ScipyOptimizer()
        new_params, score = optimizer.optimize_parameters(model_factory,
                             initial_parameters, judge,
                             data_predictor, target_data)
        error_message = "Expected float, got %s %s" % (type(score), score)
        nose.tools.ok_(type(score) is types.FloatType, error_message)
        error_message = "Expected ParameterSet, got %s" % new_params
        nose.tools.ok_(type(new_params) is type(initial_parameters),
                       error_message)
        nose.tools.ok_(abs(0.0 - score) < EPSILON,
                       "Expected score = 2.0, got %s %s" % (type(score), score))
        nose.tools.ok_(abs(2.0 - new_params.get_parameter('a')) < EPSILON,
                       "Expected optimal a = 2.0, got %s" % new_params)
        nose.tools.ok_(abs(5.0 - new_params.get_parameter('b')) < EPSILON,
                       "Expected optimal a = 5.0, got %s" % new_params)
        return
