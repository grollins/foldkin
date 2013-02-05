import foldkin.base
import numpy
from sklearn.metrics import mean_squared_error

class QuadFitModelFactory(foldkin.base.model_factory.ModelFactory):
    """QuadFitModelFactory creates a QuadFitModel."""
    def __init__(self, id_list, feature_array):
        self.id_list = id_list
        self.feature_array = feature_array

    def create_model(self, parameter_set):
        return QuadFitModel(parameter_set, self.id_list, self.feature_array)


class QuadFitModel(foldkin.base.model.Model):
    def __init__(self, parameter_set, id_list, feature_array):
        self.parameter_set = parameter_set
        self.id_list = id_list
        self.feature_array = feature_array

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_id_list(self):
        return self.id_list

    def get_feature_array(self):
        return self.feature_array

class QuadFitPrediction(foldkin.base.prediction.Prediction):
    """docstring for QuadFitPrediction"""
    def __init__(self, y_array, id_list):
        super(QuadFitPrediction, self).__init__()
        self.y_array = y_array
        self.id_list = id_list

    def __iter__(self):
        if self.id_list is None:
            for this_y in self.y_array:
                yield '', this_y
        else:
            for this_id, this_y in zip(self.id_list, self.y_array):
                yield this_id, this_y

    def as_array(self):
        return self.y_array


class QuadFitDataPredictor(foldkin.base.data_predictor.DataPredictor):
    """docstring for QuadFitPredictor"""
    def __init__(self):
        super(QuadFitDataPredictor, self).__init__()
        self.prediction_factory = QuadFitPrediction

    def predict_data(self, model):
        feature_array = model.get_feature_array()
        x = model.get_parameter('x')
        y0 = model.get_parameter('y0')
        y1 = model.get_parameter('y1')
        y2 = model.get_parameter('y2')
        prediction_array = y0 + y1*(feature_array - x) + y2*(feature_array - x)**2
        return self.prediction_factory(prediction_array, model.get_id_list())


class QuadFitParameterSet(foldkin.base.parameter_set.ParameterSet):
    """QuadFitParameterSet has two parameters, a and b."""
    def __init__(self):
        super(QuadFitParameterSet, self).__init__()
        self.parameter_dict = {'x':2.0, 'y0':4.0, 'y1':0.1, 'y2':-0.1}
        self.bounds_dict = {'x':(None, None),
                            'y0':(None, None),
                            'y1':(None, None),
                            'y2':(None, None)}

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
        """Array format: [x, y0, y1, y2]"""
        x = self.parameter_dict['x']
        y0 = self.parameter_dict['y0']
        y1 = self.parameter_dict['y1']
        y2 = self.parameter_dict['y2']
        return numpy.array([x, y0, y1, y2])

    def as_array_for_scipy_optimizer(self):
        y0 = self.parameter_dict['y0']
        y1 = self.parameter_dict['y1']
        y2 = self.parameter_dict['y2']
        return numpy.array([y0, y1, y2])

    def update_from_array(self, parameter_array):
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('y0', parameter_array[0])
        self.set_parameter('y1', parameter_array[1])
        self.set_parameter('y2', parameter_array[2])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self):
        y0_bounds = self.bounds_dict['y0']
        y1_bounds = self.bounds_dict['y1']
        y2_bounds = self.bounds_dict['y2']
        bounds = [y0_bounds, y1_bounds, y2_bounds]
        return bounds
