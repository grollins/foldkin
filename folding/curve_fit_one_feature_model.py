import numpy
from sklearn.metrics import mean_squared_error
from model_factory import ModelFactory
from model import Model
from data_predictor import DataPredictor
from target_data import TargetData
from parameter_set import ParameterSet
from judge import Judge
from prediction import Prediction

class CurveFitOneFeatureModelFactory(ModelFactory):
    """CurveFitOneFeatureModelFactory creates a CurveFitOneFeatureModel."""
    def create_model(self, parameter_set):
        return CurveFitOneFeatureModel(parameter_set)


class CurveFitOneFeatureModel(Model):
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


class CurveFitOneFeaturePrediction(Prediction):
    """docstring for CurveFitOneFeaturePrediction"""
    def __init__(self, y_array):
        super(CurveFitOneFeaturePrediction, self).__init__()
        self.y_array = y_array

    def as_array(self):
        return self.y_array


class CurveFitOneFeatureDataPredictor(DataPredictor):
    """docstring for CurveFitOneFeaturePredictor"""
    def __init__(self):
        super(CurveFitOneFeatureDataPredictor, self).__init__()
        self.prediction_factory = CurveFitOneFeaturePrediction

    def predict_data(self, model, feature_array):
        a = model.get_parameter('a')
        b = model.get_parameter('b')
        c = model.get_parameter('c')
        return self.prediction_factory(a * feature_array**b + c)


class CurveFitOneFeatureTargetData(TargetData):
    """docstring for CurveFitOneFeatureTargetData"""
    def __init__(self):
        super(CurveFitOneFeatureTargetData, self).__init__()

    def load_data(self):
        self.feature = numpy.arange(0.0, 10.0, 0.1)
        self.target = 2.0 * self.feature + 5.0

    def get_feature(self):
        return self.feature

    def get_target(self):
        return self.target


class CurveFitOneFeatureParameterSet(ParameterSet):
    """CurveFitOneFeatureParameterSet has two parameters, a and b."""
    def __init__(self):
        super(CurveFitOneFeatureParameterSet, self).__init__()
        self.parameters = {'a':0.0, 'b':1.0, 'c':0.0}
        self.bounds_dict = {'a':(None, None),
                            'b':(None, None),
                            'c':(None, None),}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def set_parameter(self, parameter_name, parameter_value):
        if self.parameters.has_key(parameter_name):
            self.parameters[parameter_name] = parameter_value
        else:
            print "Unexpected parameter", parameter_name

    def get_parameter(self, parameter_name):
        return self.parameters[parameter_name]

    def as_array(self):
        """Array format: [a, b, c]"""
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return numpy.array([a, b, c])

    def update_from_array(self, parameter_array):
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('a', parameter_array[0])
        self.set_parameter('b', parameter_array[1])
        self.set_parameter('c', parameter_array[2])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        a_bounds = self.bounds_dict['a']
        b_bounds = self.bounds_dict['b']
        c_bounds = self.bounds_dict['c']
        bounds = [a_bounds, b_bounds, c_bounds]
        return bounds


class CurveFitOneFeatureJudge(Judge):
    """docstring for CurveFitOneFeatureJudge"""
    def __init__(self):
        super(CurveFitOneFeatureJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model, feature_array)
        prediction_array = prediction.as_array()
        return mean_squared_error(target_array, prediction_array), prediction
