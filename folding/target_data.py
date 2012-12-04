import abc

class TargetData(object):
    """docstring for TargetData"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load_data(self):
        return


class SimpleTargetData(TargetData):
    """docstring for SimpleTargetData"""
    def __init__(self):
        super(SimpleTargetData, self).__init__()

    def load_data(self):
        return

        