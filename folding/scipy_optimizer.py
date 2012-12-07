from parameter_optimizer import ParameterOptimizer
import scipy.optimize

class ScipyOptimizer(ParameterOptimizer):
    """docstring for ScipyOptimizer"""
    def __init__(self):
        super(ScipyOptimizer, self).__init__()
        self.optimization_fcn = scipy.optimize.fmin_l_bfgs_b

    def optimize_parameters(self, score_fcn, parameter_set):
        bounds = parameter_set.get_parameter_bounds()
        results = self.optimization_fcn(score_fcn, x0=parameter_set.as_array(),
                                        bounds=bounds, approx_grad=1)
        print results
        optimal_parameter_array = results[0]
        parameter_set.update_from_array(optimal_parameter_array)
        score = float(results[1])
        return parameter_set, score

