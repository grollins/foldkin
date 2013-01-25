import numpy
from foldkin.base.parameter_set import ParameterSet

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

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        return self.parameter_dict[param_name]

    def as_array(self):
        log_K = self.get_parameter('log_K')
        log_alpha = self.get_parameter('log_alpha')
        log_epsilon = self.get_parameter('log_epsilon')
        log_k1 = self.get_parameter('log_k1')
        N = self.get_parameter('N')
        return numpy.array([log_K, log_alpha, log_epsilon, log_k1, N])

    def as_array_for_scipy_optimizer(self):
        log_K = self.get_parameter('log_K')
        log_alpha = self.get_parameter('log_alpha')
        log_epsilon = self.get_parameter('log_epsilon')
        log_k1 = self.get_parameter('log_k1')
        return numpy.array([log_K, log_alpha, log_epsilon, log_k1])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K, log_alpha, log_epsilon, log_k1
           N will not be updated
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K', parameter_array[0])
        self.set_parameter('log_alpha', parameter_array[1])
        self.set_parameter('log_epsilon', parameter_array[2])
        self.set_parameter('log_k1', parameter_array[3])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        '''N will not be updated'''
        log_K_bounds = self.bounds_dict['log_K']
        log_alpha_bounds = self.bounds_dict['log_alpha']
        log_epsilon_bounds = self.bounds_dict['log_epsilon']
        log_k1_bounds = self.bounds_dict['log_k1']
        bounds = [log_K_bounds, log_alpha_bounds, log_epsilon_bounds,
                  log_k1_bounds]
        return bounds


class TemperatureDependenceParameterSet(ParameterSet):
    """docstring for TemperatureDependenceParameterSet"""
    def __init__(self):
        super(TemperatureDependenceParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'u':0.0, 'g':-0.5,
                               'z':1e2, 'epsilon':-1.5,
                               'log_k1':5.0,
                               'beta':1./(0.002*300)}
        self.bounds_dict = {'u':(None, None),
                            'g':(None, None),
                            'z':(None, None),
                            'epsilon':(None, None),
                            'log_k1':(None, None),
                            'beta':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in self.parameter_dict.keys():
            return self.parameter_dict[param_name]
        elif param_name == 'log_K':
            return self.compute_log_K()
        elif param_name == 'log_alpha':
            return self.compute_log_alpha()
        elif param_name == 'log_epsilon':
            return self.compute_log_E()
        else:
            assert False, "No such parameter: %s" % param_name

    def compute_log_K(self):
        u = self.parameter_dict['u']
        z = self.parameter_dict['z']
        beta = self.parameter_dict['beta']
        K = numpy.exp(-u * beta) / z
        return numpy.log10(K)

    def compute_log_alpha(self):
        g = self.parameter_dict['g']
        beta = self.parameter_dict['beta']
        alpha = numpy.exp(-g * beta)
        return numpy.log10(alpha)

    def compute_log_E(self):
        epsilon = self.parameter_dict['epsilon']
        beta = self.parameter_dict['beta']
        E = numpy.exp(-epsilon * beta)
        return numpy.log10(E)

    def as_array(self):
        u = self.get_parameter('u')
        g = self.get_parameter('g')
        z = self.get_parameter('z')
        epsilon = self.get_parameter('epsilon')
        log_k1 = self.get_parameter('log_k1')
        beta = self.get_parameter('beta')
        N = self.get_parameter('N')
        return numpy.array([u, g, z, epsilon, log_k1, beta, N])

    def as_array_for_scipy_optimizer(self):
        u = self.get_parameter('u')
        g = self.get_parameter('g')
        z = self.get_parameter('z')
        epsilon = self.get_parameter('epsilon')
        log_k1 = self.get_parameter('log_k1')
        return numpy.array([u, g, z, epsilon, log_k1])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           u, g, z, epsilon, log_k1
           N and beta will not be set this way
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('u', parameter_array[0])
        self.set_parameter('g', parameter_array[1])
        self.set_parameter('z', parameter_array[2])
        self.set_parameter('epsilon', parameter_array[3])
        self.set_parameter('log_k1', parameter_array[4])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        '''N and beta will not be varied, so no bounds'''
        u_bounds = self.bounds_dict['u']
        g_bounds = self.bounds_dict['g']
        z_bounds = self.bounds_dict['z']
        epsilon_bounds = self.bounds_dict['epsilon']
        log_k1_bounds = self.bounds_dict['log_k1']
        bounds = [u_bounds, g_bounds, z_bounds, epsilon_bounds,
                  log_k1_bounds]
        return bounds


