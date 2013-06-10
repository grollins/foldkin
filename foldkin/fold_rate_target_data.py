import os
import pandas
import numpy
from copy import deepcopy
from foldkin.base.target_data import TargetData
from foldkin.util import boltz_k

class SingleFoldRateTargetData(TargetData):
    """SingleFoldRateTargetData reads from a table that looks like this:
    N,L,aco,rco,coc1,coc2,logkf,name,fold,pdb
    3,73,12.96,0.18,0.78,17.14,5.516,a3D,a,2A3D
    7,98,35.88,0.37,2.18,203.61,-0.617,AcP,ab,1APS
    7,98,34.30,0.35,1.94,179.04,0.365,AcP_common,ab,2ACY
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
        data_table = pandas.read_csv(data_file, index_col=7, header=0)
        self.feature = data_table.get_value(protein_name, 'N')
        self.exp_rate = data_table.get_value(protein_name, 'logkf')
        self.name = protein_name
        self.fold = data_table.get_value(protein_name, 'fold')

    def get_feature(self):
        return numpy.array([self.feature])

    def get_target(self):
        return numpy.array([self.exp_rate])

    def get_notes(self):
        return [(self.name,), (self.fold,)]

    def to_data_frame(self):
        d = {'feature':[self.feature,], 'logkf':[self.exp_rate,],
             'fold':[self.fold,]}
        df = pandas.DataFrame(d, index=[self.name,])
        return df


class FoldRateCollectionTargetData(TargetData):
    """FoldRateCollectionTargetData reads from a table that looks like this:
    N,L,aco,rco,coc1,coc2,logkf,name,fold,pdb
    3,73,12.96,0.18,0.78,17.14,5.516,a3D,a,2A3D
    7,98,35.88,0.37,2.18,203.61,-0.617,AcP,ab,1APS
    7,98,34.30,0.35,1.94,179.04,0.365,AcP_common,ab,2ACY
    .
    .
    .
    """
    def __init__(self):
        super(FoldRateCollectionTargetData, self).__init__()
        self.feature = None
        self.exp_rates = None

    def __len__(self):
        return len(self.feature)

    def __iter__(self):
        for i in xrange(len(self.feature)):
            yield (self.feature[i], self.exp_rates[i])

    def __str__(self):
        return str(self.to_data_frame())

    def iter_feature(self):
        for feature, exp_rate in self:
            yield feature, exp_rate

    def has_element(self, element):
        """Expecting element to be (feature, exp_rate) tuple."""
        feature_being_searched_for = element[0]
        exp_rate_that_goes_with_that_feature = element[1]
        is_found = False
        for i in xrange(len(self.feature)):
            stop_condition1 = (self.feature[i] == feature_being_searched_for)
            stop_condition2 = (self.exp_rates[i] == exp_rate_that_goes_with_that_feature)
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
            raise ValueError
        data_file = os.path.expanduser("~/Dropbox/python/kinetic_db/simple_table.txt")
        data_table = pandas.read_csv(data_file, index_col=7, header=0)
        self.feature = numpy.array(data_table[feature], numpy.float32)
        self.exp_rates = numpy.array(data_table['logkf'], numpy.float32)
        self.names = numpy.array(data_table.index.tolist(), numpy.str)
        self.folds = numpy.array(data_table['fold'], numpy.str)
        self.pdb_ids = numpy.array(data_table['pdb'], numpy.str)
        return

    def get_feature(self):
        return self.feature

    def get_target(self):
        return self.exp_rates

    def get_pdb_ids(self):
        return self.pdb_ids

    def get_notes(self):
        return [self.names, self.folds, self.pdb_ids]

    def make_copy_from_selection(self, inds):
        my_clone = deepcopy(self)
        my_clone.feature = my_clone.feature[inds]
        my_clone.exp_rates = my_clone.exp_rates[inds]
        my_clone.names = my_clone.names[inds]
        my_clone.folds = my_clone.folds[inds]
        my_clone.pdb_ids = my_clone.pdb_ids[inds]
        return my_clone

    def to_data_frame(self):
        d = {'feature':self.feature, 'logkf':self.exp_rates, 'fold':self.folds,
             'pdb_id':self.pdb_ids}
        df = pandas.DataFrame(d, index=self.names)
        return df


class TemperatureDependenceTargetData(TargetData):
    """docstring for TemperatureDependenceTargetData"""
    def __init__(self):
        super(TemperatureDependenceTargetData, self).__init__()
        self.feature = None
        self.exp_rates = None
        self.data_table = None

    def __len__(self):
        return len(self.feature)

    def load_data(self, protein_name, folding_or_unfolding_data,
                  arrhenius_dict=None):
        if arrhenius_dict:
            pass
        else:
            from kinetic_db.arrhenius import arrhenius_dict
        self.name = protein_name
        arrhenius_data = arrhenius_dict[self.name]
        fold_data_file, unfold_data_file, N, avg_ss_length = arrhenius_data
        self.N = N

        if folding_or_unfolding_data == 'fold':
            fold_data_table = pandas.read_csv(fold_data_file,
                                          names=('1000/T', 'logk'))
            fold_data_table['T'] = 1./(fold_data_table['1000/T'] / 1000)
            fold_data_table['beta'] = 1./(fold_data_table['T'] * boltz_k)
            self.feature = numpy.array(fold_data_table['beta'], numpy.float32)
            self.exp_rates = numpy.array(fold_data_table['logk'], numpy.float32)
            self.data_table = fold_data_table
        elif folding_or_unfolding_data == 'unfold':
            unfold_data_table = pandas.read_csv(unfold_data_file,
                                                names=('1000/T', 'logk'))
            unfold_data_table['T'] = 1./(unfold_data_table['1000/T'] / 1000)
            unfold_data_table['beta'] = 1./(unfold_data_table['T'] * boltz_k)
            self.feature = numpy.array(unfold_data_table['beta'], numpy.float32)
            self.exp_rates = numpy.array(unfold_data_table['logk'], numpy.float32)
            self.data_table = unfold_data_table
        else:
            print "Unknown data set", folding_or_unfolding_data

        return

    def get_feature(self):
        return self.feature

    def get_target(self):
        return self.exp_rates

    def get_N(self):
        return self.N

    def get_feature_at_max_target(self):
        ind = numpy.argmax(self.exp_rates)
        return self.feature[ind]

    def get_max_target(self):
        return numpy.max(self.exp_rates)

    def get_median_target(self):
        median_value = numpy.median(self.exp_rates)
        delta = numpy.abs(self.exp_rates - median_value)
        ind = numpy.argmin(delta)
        beta = self.feature[ind]
        return beta, median_value

    def make_copy_from_selection(self, inds):
        my_clone = deepcopy(self)
        my_clone.feature = my_clone.feature[inds]
        my_clone.exp_rates = my_clone.exp_rates[inds]
        my_clone.data_table = my_clone.data_table.ix[inds]
        return my_clone

    def to_data_frame(self):
        return self.data_table
