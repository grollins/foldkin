import base
import numpy

class SimpleModelFactory(base.model_factory.ModelFactory):
    """SimpleModelFactory creates a SimpleModel."""
    def create_model(self, parameter_set):
        return SimpleModel(parameter_set)


class SimpleModel(base.model.Model):
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


class SimplePrediction(base.prediction.Prediction):
    """docstring for SimplePrediction"""
    def __init__(self, y):
        super(SimplePrediction, self).__init__()
        self.y = y

    def as_array(self):
        return numpy.array([self.y])


class SimpleDataPredictor(base.data_predictor.DataPredictor):
    """docstring for SimplePredictor"""
    def __init__(self):
        super(SimpleDataPredictor, self).__init__()
        self.prediction_factory = SimplePrediction

    def predict_data(self, model, feature):
        x = model.get_parameter('x')
        return self.prediction_factory( (x - 3)**2 + 2 )


class SimpleTargetData(base.target_data.TargetData):
    """docstring for SimpleTargetData"""
    def __init__(self):
        super(SimpleTargetData, self).__init__()

    def load_data(self):
        return

    def get_feature(self):
        return

    def get_target(self):
        return

    def get_notes(self):
        return []


class SimpleParameterSet(base.parameter_set.ParameterSet):
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


class SimpleJudge(base.judge.Judge):
    """docstring for SimpleJudge"""
    def __init__(self):
        super(SimpleJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        prediction = data_predictor.predict_data(model, None)
        prediction_array = prediction.as_array()
        return prediction_array[0], prediction
