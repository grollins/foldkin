import abc

class DataPredictor(object):
    """DataPredictor is an abstract class"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_data(self, model):
        return


class SimpleDataPredictor(DataPredictor):
    """docstring for SimplePredictor"""
    def __init__(self):
        super(SimpleDataPredictor, self).__init__()

    def predict_data(self, model):
        x = model.get_parameter('x')
        y = (x - 3)**2 + 2
        return y
