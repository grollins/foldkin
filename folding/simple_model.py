from model_factory import ModelFactory

class SimpleModelFactory(ModelFactory):
    """SimpleModelFactory creates a SimpleModel."""
    def create_model(self, parameter_set):
        return SimpleModel(parameter_set)

class SimpleModel(object):
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)
