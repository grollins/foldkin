import abc
import numpy

class ParameterSet(object):
    """ParameterSet is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_parameter(self, parameter_name, parameter_value):
        return

    @abc.abstractmethod
    def get_parameter(self, parameter_name):
        return

    @abc.abstractmethod
    def as_array(self):
        return

    @abc.abstractmethod
    def update_from_array(self, parameter_array):
        return


class SimpleParameterSet(ParameterSet):
    """SimpleParameterSet has one parameter, x."""
    def __init__(self):
        super(SimpleParameterSet, self).__init__()
        self.x = 0.0

    def set_parameter(self, parameter_name, parameter_value):
        if parameter_name == 'x':
            self.x = parameter_value
        else:
            print "Unexpected parameter", parameter_name

    def get_parameter(self, parameter_name):
        if parameter_name == 'x':
            return self.x
        else:
            print "Unexpected parameter", parameter_name

    def as_array(self):
        """Array format: [x]"""
        return numpy.array([self.x,])

    def update_from_array(self, parameter_array):
        self.set_parameter('x', numpy.atleast_1d(parameter_array)[0])
