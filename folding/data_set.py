import abc
import os
import pandas
import numpy
from copy import deepcopy

class DataSet(object):
    """DataSet is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def as_array(self):
        return

    @abc.abstractmethod
    def get_id(self):
        return

    @abc.abstractmethod
    def load_data(self):
        return

    @abc.abstractmethod
    def sample(self, inds):
        return


class SimpleDataSet(DataSet):
    """docstring for SimpleDataSet"""
    def __init__(self, y):
        super(SimpleDataSet, self).__init__()
        self.y = y
        self.id = ''
    def as_array(self):
        return numpy.array([y])
    def get_id(self):
        return self.id
    def load_data(self):
        return
    def sample(self):
        return self


class FoldingRateSet(DataSet):
    """docstring for FoldingRateSet"""
    def __init__(self):
        self.id = 'kf'
        self.num_ss = None
        self.exp_rates = None

    def __len__(self):
        return len(self.exp_rates)

    def as_array(self):
        return self.exp_rates

    def get_id(self):
        return self.id

    def load_data(self):
        data_file = os.path.expanduser("~/Dropbox/python/kinetic_db/simple_table.txt")
        data_table = pandas.read_csv(data_file, index_col=5, header=0)
        self.num_ss = data_table['N']
        self.exp_rates = data_table['logkf']

    def sample(self, inds):
        my_copy = deepcopy(self)
        my_copy.num_ss = my_copy.num_ss[inds]
        my_copy.exp_rates = my_copy.exp_rates[inds]
        return my_copy


class T_DependentRateSet(DataSet):
    """docstring for T_DependentRateSet"""
    def __init__(self):
        self.id = 'arrhenius'

    def as_array(self):
        return numpy.array([])

    def get_id(self):
        return self.id

    def sample(self):
        return deepcopy(self)
