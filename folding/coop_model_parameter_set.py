from parameter_set import ParameterSet
import numpy

class CoopModelParameterSet(ParameterSet):
    """docstring for CoopModelParameterSet"""
    def __init__(self):
        super(CoopModelParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'K':0.1, 'alpha':2.0,
                               'epsilon':10**6.0, 'k1':10**6.0}

    def set_parameter(self, param_name, param_value):
        self.parameter_dict[param_name] = param_value

    def get_parameter(self, param_name):
        return self.parameter_dict[param_name]

    def as_array(self):
        K = self.parameter_dict['K']
        alpha = self.parameter_dict['alpha']
        epsilon = self.parameter_dict['epsilon']
        k1 = self.parameter_dict['k1']
        N = self.parameter_dict['N']
        return numpy.array([K, alpha, epsilon, k1])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           K, alpha, epsilon, k1,
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('K', parameter_array[0])
        self.set_parameter('alpha', parameter_array[1])
        self.set_parameter('epsilon', parameter_array[2])
        self.set_parameter('k1', parameter_array[3])
        # self.set_parameter('N', parameter_array[4])
