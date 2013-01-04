import base
import numpy
from sklearn.metrics import mean_squared_error

class KingsFeatureModelFactory(base.model_factory.ModelFactory):
    """KingsFeatureModelFactory creates a KingsFeatureModel."""
    def create_model(self, parameter_set):
        return KingsFeatureModel(parameter_set)


class KingsFeatureModel(base.model.Model):
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


class KingsFeaturePrediction(base.prediction.Prediction):
    """docstring for KingsFeaturePrediction"""
    def __init__(self, y_array):
        super(KingsFeaturePrediction, self).__init__()
        self.y_array = y_array

    def as_array(self):
        return self.y_array


class KingsFeatureDataPredictor(base.data_predictor.DataPredictor):
    """docstring for KingsFeaturePredictor"""
    def __init__(self):
        super(KingsFeatureDataPredictor, self).__init__()
        self.prediction_factory = KingsFeaturePrediction

    def predict_data(self, model, aco, coc1, coc2):
        a = model.get_parameter('a')
        b = model.get_parameter('b')
        c = model.get_parameter('c')
        return self.prediction_factory(-a * aco + a**2 * coc1 + b * coc2 + c)


class KingsFeatureParameterSet(base.parameter_set.ParameterSet):
    """KingsFeatureParameterSet has two parameters, a and b."""
    def __init__(self):
        super(KingsFeatureParameterSet, self).__init__()
        self.parameter_dict = {'a':0.0, 'b':1.0, 'c':0.0}
        self.bounds_dict = {'a':(None, None),
                            'b':(None, None),
                            'c':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, parameter_name, parameter_value):
        if self.parameter_dict.has_key(parameter_name):
            self.parameter_dict[parameter_name] = parameter_value
        else:
            print "Unexpected parameter", parameter_name

    def get_parameter(self, parameter_name):
        return self.parameter_dict[parameter_name]

    def as_array(self):
        """Array format: [a, b, c]"""
        a = self.parameter_dict['a']
        b = self.parameter_dict['b']
        c = self.parameter_dict['c']
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


class KingsFeatureJudge(base.judge.Judge):
    """docstring for KingsFeatureJudge"""
    def __init__(self):
        super(KingsFeatureJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature_array = target_data.get_feature()
        aco = feature_array[:,0]
        coc1 = feature_array[:,1]
        coc2 = feature_array[:,2]
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model, aco, coc1, coc2)
        prediction_array = prediction.as_array()
        error_msg = "%s %s" % (prediction_array.shape, target_array.shape)
        assert prediction_array.shape == target_array.shape, error_msg
        return mean_squared_error(target_array, prediction_array), prediction
