import numpy
from sklearn.metrics import mean_squared_error
from model_factory import ModelFactory
from model import Model
from data_predictor import DataPredictor
from target_data import TargetData
from parameter_set import ParameterSet
from judge import Judge

class LinearModelFactory(ModelFactory):
    """LinearModelFactory creates a LinearModel."""
    def create_model(self, parameter_set):
        return LinearModel(parameter_set)


class LinearModel(Model):
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


class LinearDataPredictor(DataPredictor):
    """docstring for LinearPredictor"""
    def __init__(self):
        super(LinearDataPredictor, self).__init__()

    def predict_data(self, model, feature):
        a = model.get_parameter('a')
        b = model.get_parameter('b')
        prediction = a * feature + b
        return prediction


class LinearTargetData(TargetData):
    """docstring for LinearTargetData"""
    def __init__(self):
        super(LinearTargetData, self).__init__()

    def load_data(self):
        self.feature = numpy.arange(0.0, 10.0, 0.1)
        self.target = 2.0 * self.feature + 5.0

    def get_feature(self):
        return self.feature

    def get_target(self):
        return self.target


class LinearParameterSet(ParameterSet):
    """LinearParameterSet has two parameters, a and b."""
    def __init__(self):
        super(LinearParameterSet, self).__init__()
        self.parameters = {'a':0.0, 'b':0.0}

    def set_parameter(self, parameter_name, parameter_value):
        if self.parameters.has_key(parameter_name):
            self.parameters[parameter_name] = parameter_value
        else:
            print "Unexpected parameter", parameter_name

    def get_parameter(self, parameter_name):
        return self.parameters[parameter_name]

    def as_array(self):
        """Array format: [a, b]"""
        a = self.parameters['a']
        b = self.parameters['b']
        return numpy.array([a, b])

    def update_from_array(self, parameter_array):
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('a', parameter_array[0])
        self.set_parameter('b', parameter_array[1])


class LinearJudge(Judge):
    """docstring for LinearJudge"""
    def __init__(self):
        super(LinearJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature = target_data.get_feature()
        target = target_data.get_target()
        prediction = data_predictor.predict_data(model, feature)
        return mean_squared_error(target, prediction), prediction
