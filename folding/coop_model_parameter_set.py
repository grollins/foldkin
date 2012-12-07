from parameter_set import ParameterSet
import numpy

class CoopModelParameterSet(ParameterSet):
    """docstring for CoopModelParameterSet"""
    def __init__(self):
        super(CoopModelParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'log_K':-1.0, 'log_alpha':0.3,
                               'log_epsilon':6.0, 'log_k1':6.0}
        self.bounds_dict = {'log_K':(None, None),
                            'log_alpha':(None, None),
                            'log_epsilon':(None, None),
                            'log_k1':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def set_parameter(self, param_name, param_value):
        self.parameter_dict[param_name] = param_value

    def get_parameter(self, param_name):
        return self.parameter_dict[param_name]

    def as_array(self):
        log_K = self.get_parameter('log_K')
        log_alpha = self.get_parameter('log_alpha')
        log_epsilon = self.get_parameter('log_epsilon')
        log_k1 = self.get_parameter('log_k1')
        N = self.get_parameter('N')
        return numpy.array([log_K, log_alpha, log_epsilon, log_k1, N])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K, log_alpha, log_epsilon, log_k1, N
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K', parameter_array[0])
        self.set_parameter('log_alpha', parameter_array[1])
        self.set_parameter('log_epsilon', parameter_array[2])
        self.set_parameter('log_k1', parameter_array[3])
        self.set_parameter('N', int(parameter_array[4]))

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        log_K_bounds = self.bounds_dict['log_K']
        log_alpha_bounds = self.bounds_dict['log_alpha']
        log_epsilon_bounds = self.bounds_dict['log_epsilon']
        log_k1_bounds = self.bounds_dict['log_k1']
        N = self.get_parameter('N')
        N_bounds = (N, N)
        bounds = [log_K_bounds, log_alpha_bounds, log_epsilon_bounds,
                  log_k1_bounds, N_bounds]
        return bounds
