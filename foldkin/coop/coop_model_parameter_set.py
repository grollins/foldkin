import numpy
from foldkin.base.parameter_set import ParameterSet
from foldkin.util import convert_beta_to_T


class CoopModelParameterSet(ParameterSet):
    """docstring for CoopModelParameterSet"""
    def __init__(self):
        super(CoopModelParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'log_K_ss':-1.0,
                               'log_K_ter':0.3, 'log_K_f':6.0,
                               'log_k0':5.7}
        self.bounds_dict = {'log_K_ss':(None, None),
                            'log_K_ter':(None, None),
                            'log_K_f':(None, None),
                            'log_k0':(None, None)}

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
        log_K_ss = self.get_parameter('log_K_ss')
        log_K_ter = self.get_parameter('log_K_ter')
        log_K_f = self.get_parameter('log_K_f')
        log_k0 = self.get_parameter('log_k0')
        N = self.get_parameter('N')
        return numpy.array([log_K_ss, log_K_ter, log_K_f, log_k0, N])

    def as_array_for_scipy_optimizer(self):
        log_K_ss = self.get_parameter('log_K_ss')
        log_K_ter = self.get_parameter('log_K_ter')
        log_K_f = self.get_parameter('log_K_f')
        log_k0 = self.get_parameter('log_k0')
        return numpy.array([log_K_ss, log_K_ter, log_K_f, log_k0])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K_ss, log_K_ter, log_K_f, log_k0
           N will not be updated
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K_ss', parameter_array[0])
        self.set_parameter('log_K_ter', parameter_array[1])
        self.set_parameter('log_K_f', parameter_array[2])
        self.set_parameter('log_k0', parameter_array[3])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        '''N will not be updated'''
        log_K_ss_bounds = self.bounds_dict['log_K_ss']
        log_K_ter_bounds = self.bounds_dict['log_K_ter']
        log_K_f_bounds = self.bounds_dict['log_K_f']
        log_k0_bounds = self.bounds_dict['log_k0']
        bounds = [log_K_ss_bounds, log_K_ter_bounds, log_K_f_bounds,
                  log_k0_bounds]
        return bounds


class TemperatureDependenceParameterSet(ParameterSet):
    """docstring for TemperatureDependenceParameterSet"""
    def __init__(self):
        super(TemperatureDependenceParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'H_ss':0.0, 'H_ter':-0.5,
                               'S_ss':-1e-2, 'S_ter':-1e-2,
                               'G_f':-1.5, 'log_k0':5.0,
                               'beta':1./(0.002*300)}
        self.bounds_dict = {'H_ss':(None, None),
                            'H_ter':(None, None),
                            'S_ss':(None, None),
                            'S_ter':(None, None),
                            'G_f':(None, None),
                            'log_k0':(None, None),
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
        elif param_name == 'log_K_ss':
            return self.compute_log_K_ss()
        elif param_name == 'log_K_ter':
            return self.compute_log_K_ter()
        elif param_name == 'log_K_f':
            return self.compute_log_K_f()
        else:
            assert False, "No such parameter: %s" % param_name

    def compute_log_K_ss(self):
        H_ss = self.parameter_dict['H_ss']
        S_ss = self.parameter_dict['S_ss']
        beta = self.parameter_dict['beta']
        T = convert_beta_to_T(beta)
        K_ss = numpy.exp(-(H_ss - T * S_ss) * beta)
        return numpy.log10(K_ss)

    def compute_log_K_ter(self):
        H_ter = self.parameter_dict['H_ter']
        S_ter = self.parameter_dict['S_ter']
        beta = self.parameter_dict['beta']
        T = convert_beta_to_T(beta)
        K_ter = numpy.exp(-(H_ter - T * S_ter) * beta)
        return numpy.log10(K_ter)

    def compute_log_K_f(self):
        G_f = self.parameter_dict['G_f']
        beta = self.parameter_dict['beta']
        K_f = numpy.exp(-G_f * beta)
        return numpy.log10(K_f)

    def as_array(self):
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        S_ss = self.get_parameter('S_ss')
        S_ter = self.get_parameter('S_ter')
        G_f = self.get_parameter('G_f')
        log_k0 = self.get_parameter('log_k0')
        beta = self.get_parameter('beta')
        N = self.get_parameter('N')
        return numpy.array([H_ss, H_ter, S_ss, S_ter, G_f,
                            log_k0, beta, N])

    def as_array_for_scipy_optimizer(self):
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        S_ss = self.get_parameter('S_ss')
        S_ter = self.get_parameter('S_ter')
        G_f = self.get_parameter('G_f')
        log_k0 = self.get_parameter('log_k0')
        return numpy.array([H_ss, H_ter, S_ss, S_ter, G_f, log_k0])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           H_ss, H_ter, S_ss, S_ter, G_f, log_k0
           N and beta will not be set this way
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('H_ss', parameter_array[0])
        self.set_parameter('H_ter', parameter_array[1])
        self.set_parameter('S_ss', parameter_array[2])
        self.set_parameter('S_ter', parameter_array[3])
        self.set_parameter('G_f', parameter_array[4])
        self.set_parameter('log_k0', parameter_array[5])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        '''N and beta will not be varied, so no bounds'''
        H_ss_bounds = self.bounds_dict['H_ss']
        H_ter_bounds = self.bounds_dict['H_ter']
        S_ss_bounds = self.bounds_dict['S_ss']
        S_ter_bounds = self.bounds_dict['S_ter']
        G_f_bounds = self.bounds_dict['G_f']
        log_k0_bounds = self.bounds_dict['log_k0']
        bounds = [H_ss_bounds, H_ter_bounds, S_ss_bounds, S_ter_bounds,
                  G_f_bounds, log_k0_bounds]
        return bounds


