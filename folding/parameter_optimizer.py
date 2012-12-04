import abc

class ParameterOptimizer(object):
    """ParameterOptimizer is an abstract class"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimize_parameters(self, model_factory,
                             parameter_set, judge,
                             data_predictor, target_data):
        return
