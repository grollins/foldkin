import numpy
from model_factory import ModelFactory
from model import Model
from data_predictor import DataPredictor
from target_data import TargetData
from parameter_set import ParameterSet
from judge import Judge
from prediction import Prediction

class SimpleModelFactory(ModelFactory):
    """SimpleModelFactory creates a SimpleModel."""
    def create_model(self, parameter_set):
        return SimpleModel(parameter_set)


class SimpleModel(Model):
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


class SimplePrediction(Prediction):
    """docstring for SimplePrediction"""
    def __init__(self, y):
        super(SimplePrediction, self).__init__()
        self.y = y

    def as_array(self):
        return numpy.array([self.y])


class SimpleDataPredictor(DataPredictor):
    """docstring for SimplePredictor"""
    def __init__(self):
        super(SimpleDataPredictor, self).__init__()
        self.prediction_factory = SimplePrediction

    def predict_data(self, model, feature):
        x = model.get_parameter('x')
        return self.prediction_factory( (x - 3)**2 + 2 )


class SimpleTargetData(TargetData):
    """docstring for SimpleTargetData"""
    def __init__(self):
        super(SimpleTargetData, self).__init__()

    def load_data(self):
        return

    def get_feature(self):
        return

    def get_target(self):
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

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        return

    def get_parameter_bounds(self):
        return [ (None, None) ]


class SimpleJudge(Judge):
    """docstring for SimpleJudge"""
    def __init__(self):
        super(SimpleJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        prediction = data_predictor.predict_data(model, None)
        prediction_array = prediction.as_array()
        return prediction_array[0], prediction
