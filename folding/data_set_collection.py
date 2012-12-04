import abc
from data_set import FoldingRateSet, T_DependentRateSet

class DataSetCollection(object):
    """DataSetCollection is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_data_set(self, data_set_name):
        return

    @abc.abstractmethod
    def set_data_set(self, data_set_name, data_set):
        return


class FoldingDataSetCollection(DataSetCollection):
    """docstring for DataSetCollection"""
    def __init__(self):
        super(DataSetCollection, self).__init__()
        self.data_set_class_dict = {'kf':FoldingRateSet,
                                    'arrhenius':T_DependentRateSet}
        self.data = {}

    def __iter__(self):
        for data_set in self.data.values():
            yield data_set

    def load_data(self, requested_data_keys):
        for k in requested_data_keys:
            this_data_set_factory = self.data_set_class_dict[k]
            this_data_set = this_data_set_factory()
            this_data_set.load_data()
            self.data[k] = this_data_set

    def get_data_set(self, data_set_name):
        return self.data[data_set_name]

    def set_data_set(self, data_set_name, data_set):
        self.data[data_set_name] = data_set
