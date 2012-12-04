from folding.scipy_optimizer import ScipyOptimizer
import simple_model
import nose.tools
from types import FloatType

EPSILON = 1e-3


@nose.tools.istest
class TestSimpleOptimization(object):
    @nose.tools.istest
    def return_correct_score_and_optimal_parameter_value(self):
        '''This example optimizes y = (x-3)^2 + 2 to find
            the value of x that minimizes y.
        '''
        model_factory = simple_model.SimpleModelFactory()
        initial_parameters = simple_model.SimpleParameterSet()
        initial_parameters.set_parameter('x', 0.0)
        judge = simple_model.SimpleJudge()
        data_predictor = simple_model.SimpleDataPredictor()
        target_data = simple_model.SimpleTargetData()
        target_data.load_data()
        optimizer = ScipyOptimizer()
        new_params, score = optimizer.optimize_parameters(model_factory,
                             initial_parameters, judge,
                             data_predictor, target_data)
        error_message = "Expected float, got %s %s" % (type(score), score)
        nose.tools.ok_(type(score) is FloatType, error_message)
        error_message = "Expected ParameterSet, got %s" % new_params
        nose.tools.ok_(type(new_params) is type(initial_parameters),
                       error_message)
        nose.tools.ok_(abs(2.0 - score) < EPSILON,
                       "Expected score = 2.0, got %s %s" % (type(score), score))
        nose.tools.ok_(abs(3.0 - new_params.get_parameter('x')) < EPSILON,
                       "Expected optimal x = 3.0, got %s" % new_params)
        return
