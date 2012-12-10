import os
import pandas
import numpy
from copy import deepcopy
import base.target_data

class SingleFoldRateTargetData(base.target_data.TargetData):
    """SingleFoldRateTargetData reads from a table that looks like this:
    N,L,aco,rco,logkf,name
    3,73,12.96,0.18,5.516,a3D
    7,98,35.88,0.37,-0.617,AcP
    7,98,34.30,0.35,0.365,AcP_common
    .
    .
    .
    """
    def __init__(self):
        super(SingleFoldRateTargetData, self).__init__()
        self.num_ss = None
        self.exp_rate = None

    def load_data(self, protein_name):
        data_file = os.path.expanduser("~/Dropbox/python/kinetic_db/simple_table.txt")
        data_table = pandas.read_csv(data_file, index_col=5, header=0)
        self.num_ss = data_table.get_value(protein_name, 'N')
        self.exp_rate = data_table.get_value(protein_name, 'logkf')
        self.name = protein_name
        self.fold = data_table.get_value(protein_name, 'fold')

    def get_feature(self):
        return numpy.array([self.num_ss])

    def get_target(self):
        return numpy.array([self.exp_rate])

    def get_notes(self):
        return [(self.name,), (self.fold,)]

class FoldRateCollectionTargetData(base.target_data.TargetData):
    """FoldRateCollectionTargetData reads from a table that looks like this:
    N,L,aco,rco,logkf,name
    3,73,12.96,0.18,5.516,a3D
    7,98,35.88,0.37,-0.617,AcP
    7,98,34.30,0.35,0.365,AcP_common
    .
    .
    .
    """
    def __init__(self):
        super(FoldRateCollectionTargetData, self).__init__()
        self.feature = None
        self.exp_rate = None

    def __len__(self):
        return len(self.feature)

    def __iter__(self):
        for i in xrange(len(self.feature)):
            yield (self.feature[i], self.exp_rate[i])

    def iter_feature(self):
        for i in xrange(len(self.feature)):
            yield self.feature[i]

    def has_element(self, element):
        """Expecting element to be (feature, exp_rate) tuple."""
        feature_being_searched_for = element[0]
        exp_rate_that_goes_with_that_feature = element[1]
        is_found = False
        for i in xrange(len(self.feature)):
            stop_condition1 = (self.feature[i] == feature_being_searched_for)
            stop_condition2 = (self.exp_rate[i] == exp_rate_that_goes_with_that_feature)
            if stop_condition1 and stop_condition2:
               is_found = True
               break
            else:
                continue
        return is_found

    def load_data(self, feature):
        if feature in ['N', 'L', 'aco', 'rco']:
            pass
        else:
            print "Unrecognized data feature:", feature
            return
        data_file = os.path.expanduser("~/Dropbox/python/kinetic_db/simple_table.txt")
        data_table = pandas.read_csv(data_file, index_col=5, header=0)
        self.feature = numpy.array(data_table[feature], numpy.float32)
        self.exp_rate = numpy.array(data_table['logkf'], numpy.float32)
        self.names = numpy.array(data_table.index.tolist(), numpy.str)
        self.folds = numpy.array(data_table['fold'], numpy.str)
        return

    def get_feature(self):
        return numpy.array(self.feature)

    def get_target(self):
        return numpy.array(self.exp_rate)

    def get_notes(self):
        return [self.names, self.folds]

    def make_copy_from_selection(self, inds):
        my_clone = deepcopy(self)
        my_clone.feature = my_clone.feature[inds]
        my_clone.exp_rate = my_clone.exp_rate[inds]
        my_clone.names = my_clone.names[inds]
        my_clone.folds = my_clone.folds[inds]
        return my_clone
