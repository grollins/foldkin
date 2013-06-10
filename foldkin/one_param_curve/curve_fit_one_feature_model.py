import foldkin.base
import numpy
from sklearn.metrics import mean_squared_error

class CurveFitOneFeatureModelFactory(foldkin.base.model_factory.ModelFactory):
    """CurveFitOneFeatureModelFactory creates a CurveFitOneFeatureModel."""
    def __init__(self, id_list):
        self.id_list = id_list

    def create_model(self, parameter_set):
        return CurveFitOneFeatureModel(parameter_set, self.id_list)


class CurveFitOneFeatureModel(foldkin.base.model.Model):
    def __init__(self, parameter_set, id_list):
        self.parameter_set = parameter_set
        self.id_list = id_list

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_id_list(self):
        return self.id_list


class CurveFitOneFeaturePrediction(foldkin.base.prediction.Prediction):
    """docstring for CurveFitOneFeaturePrediction"""
    def __init__(self, y_array, id_list):
        super(CurveFitOneFeaturePrediction, self).__init__()
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


class CurveFitOneFeatureDataPredictor(foldkin.base.data_predictor.DataPredictor):
    """docstring for CurveFitOneFeaturePredictor"""
    def __init__(self):
        super(CurveFitOneFeatureDataPredictor, self).__init__()
        self.prediction_factory = CurveFitOneFeaturePrediction

    def predict_data(self, model, feature_array):
        a = model.get_parameter('a')
        b = model.get_parameter('b')
        c = model.get_parameter('c')
        prediction_array = a * (feature_array**b) + c
        return self.prediction_factory(prediction_array, model.get_id_list())


class CurveFitOneFeatureTargetData(foldkin.base.target_data.TargetData):
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

    def get_id_list(self):
        return [''] * len(self.feature)

    def to_data_frame(self):
        d = {'feature':self.feature, 'target':self.target}
        df = pandas.DataFrame(d, index=range(len(self.feature)))
        return df


class CurveFitOneFeatureParameterSet(foldkin.base.parameter_set.ParameterSet):
    """CurveFitOneFeatureParameterSet has two parameters, a and b."""
    def __init__(self):
        super(CurveFitOneFeatureParameterSet, self).__init__()
        self.parameter_dict = {'a':0.0, 'b':1.0, 'c':0.0}
        self.bounds_dict = {'a':(None, None),
                            'b':(None, None),
                            'c':(None, None),}

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

    def as_array_for_scipy_optimizer(self):
        return self.as_array()

    def update_from_array(self, parameter_array):
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('a', parameter_array[0])
        self.set_parameter('b', parameter_array[1])
        self.set_parameter('c', parameter_array[2])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        a_bounds = self.bounds_dict['a']
        b_bounds = self.bounds_dict['b']
        c_bounds = self.bounds_dict['c']
        bounds = [a_bounds, b_bounds, c_bounds]
        return bounds


class CurveFitOneFeatureJudge(foldkin.base.judge.Judge):
    """docstring for CurveFitOneFeatureJudge"""
    def __init__(self):
        super(CurveFitOneFeatureJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data,
                         noisy=False):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model, feature_array)
        prediction_array = prediction.as_array()

        score = mean_squared_error(target_array, prediction_array)
        if noisy:
            print score
            for i,p in enumerate(prediction):
                print p, prediction_array[i], target_array[i]
        return mean_squared_error(target_array, prediction_array), prediction


class CurveFitR2Judge(foldkin.base.judge.Judge):
    """docstring for CurveFitR2Judge"""
    def __init__(self):
        super(CurveFitR2Judge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data,
                         noisy=False):
        feature_array = target_data.get_feature()
        target_array = target_data.get_target()
        prediction = data_predictor.predict_data(model, feature_array)
        prediction_array = prediction.as_array()

        cc = numpy.corrcoef(target_array, prediction_array)[0,1]
        R2 = cc**2
        score = R2
        if noisy:
            print score
            for i,p in enumerate(prediction):
                print p, prediction_array[i], target_array[i]
        return score, prediction
