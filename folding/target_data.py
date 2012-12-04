import abc

class TargetData(object):
    """docstring for TargetData"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load_data(self):
        return

        