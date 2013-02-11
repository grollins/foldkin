import scipy.optimize
import base.parameter_optimizer

class ScipyOptimizer(base.parameter_optimizer.ParameterOptimizer):
    """docstring for ScipyOptimizer"""
    def __init__(self, factr=1e6, pgtol=1e-5, epsilon=1e-8, maxfun=100):
        super(ScipyOptimizer, self).__init__()
        self.optimization_fcn = scipy.optimize.fmin_l_bfgs_b
        self.factr = factr
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.maxfun = maxfun

    def optimize_parameters(self, score_fcn, parameter_set, noisy=False):
        bounds = parameter_set.get_parameter_bounds_list()
        parameter_array = parameter_set.as_array_for_scipy_optimizer()
        results = self.optimization_fcn(score_fcn, x0=parameter_array,
                                        bounds=bounds, approx_grad=1,
                                        factr=self.factr, pgtol=self.pgtol,
                                        epsilon=self.epsilon, maxfun=self.maxfun)
        if noisy:
            print results
        optimal_parameter_array = results[0]
        parameter_set.update_from_array(optimal_parameter_array)
        score = float(results[1])
        num_iterations = int(results[2]['funcalls'])
        return parameter_set, score, num_iterations


