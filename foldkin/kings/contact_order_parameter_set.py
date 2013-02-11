import numpy
from foldkin.base.parameter_set import ParameterSet

class ContactOrderParameterSet(ParameterSet):
    """docstring for ContactOrderParameterSet"""
    def __init__(self):
        super(ContactOrderParameterSet, self).__init__()
        self.parameter_dict = {'logk0':6.0, 'gamma':1.0}
        self.bounds_dict = {'logk0':(None, None),
                            'gamma':(None, None)}

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        return self.parameter_dict[param_name]

    def as_array(self):
        logk0 = self.get_parameter('logk0')
        gamma = self.get_parameter('gamma')
        return numpy.array([logk0, gamma])

    def as_array_for_scipy_optimizer(self):
        return self.as_array()

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           logk0, gamma
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('logk0', parameter_array[0])
        self.set_parameter('gamma', parameter_array[1])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        logk0_bounds = self.bounds_dict['logk0']
        gamma_bounds = self.bounds_dict['gamma']
        bounds = [logk0_bounds, gamma_bounds]
        return bounds
