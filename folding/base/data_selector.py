import abc

class DataSelector(object):
    """DataSelector is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample_data(self, data_set):
        return
