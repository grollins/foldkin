from folding.scipy_optimizer import ScipyOptimizer
from folding.parameter_set import SimpleParameterSet
from folding.simple_model import SimpleModelFactory, SimpleModel
from folding.data_predictor import SimpleDataPredictor
from folding.target_data import SimpleTargetData
from folding.judge import SimpleJudge
import numpy
import nose.tools
import types

EPSILON = 1e-3


@nose.tools.istest
class TestScipyOptimizer(object):
    @nose.tools.istest
    def return_correct_score_and_optimal_parameter_value(self):
        '''This example optimizes y = (x-3)^2 + 2 to find
            the value of x that minimizes y.
        '''
        model_factory = SimpleModelFactory()
        initial_parameters = SimpleParameterSet()
        initial_parameters.set_parameter('x', 0.0)
        judge = SimpleJudge()
        data_predictor = SimpleDataPredictor()
        target_data = SimpleTargetData()
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
        nose.tools.ok_(abs(2.0 - score) < EPSILON,
                       "Expected score = 2.0, got %s %s" % (type(score), score))
        nose.tools.ok_(abs(3.0 - new_params.get_parameter('x')) < EPSILON,
                       "Expected optimal x = 3.0, got %s" % new_params)
        return
