from parameter_optimizer import ParameterOptimizer
import scipy.optimize

class ScipyOptimizer(ParameterOptimizer):
    """docstring for ScipyOptimizer"""
    def __init__(self):
        super(ScipyOptimizer, self).__init__()
        self.optimization_fcn = scipy.optimize.fmin_powell

    def optimize_parameters(self, model_factory, parameter_set,
                             judge, data_predictor, target_data):
        def f(current_parameter_array, *args):
            # model_factory = args[0]
            # parameter_set_factory = args[1]
            # score_keeper = args[2]
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score = judge.judge_prediction(current_model, data_predictor,
                                           target_data)
            return score
        # extra_args = (model_factory, parameter_set)
        extra_args = ()
        results = self.optimization_fcn(func=f,
                                        x0=parameter_set.as_array(),
                                        args=extra_args, full_output=1)
        print results
        optimal_parameter_array = results[0]
        parameter_set.update_from_array(optimal_parameter_array)
        score = float(results[1])
        return parameter_set, score
