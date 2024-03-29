import os
import pandas
import numpy
from copy import deepcopy
from foldkin.base.target_data import TargetData

class SingleContactOrderTargetData(TargetData):
    """SingleContactOrderTargetData reads from a table that looks like this:
    N,L,aco,rco,coc1,coc2,logkf,name,fold,pdb
    3,73,12.96,0.18,0.78,17.14,5.516,a3D,a,2A3D
    7,98,35.88,0.37,2.18,203.61,-0.617,AcP,ab,1APS
    7,98,34.30,0.35,1.94,179.04,0.365,AcP_common,ab,2ACY
    .
    .
    .
    """
    def __init__(self):
        super(SingleContactOrderTargetData, self).__init__()
        self.num_ss = None
        self.exp_rate = None

    def load_data(self, protein_name, data_path="foldkin/kings/data_for_kings_fit.txt"):
        data_file = os.path.expanduser(data_path)
        data_table = pandas.read_csv(data_file, index_col=7, header=0)
        self.aco = data_table.get_value(protein_name, 'aco')
        self.exp_rate = data_table.get_value(protein_name, 'logkf')
        self.name = protein_name
        self.fold = data_table.get_value(protein_name, 'fold')
        self.pdb_id = data_table.get_value(protein_name, 'pdb')

    def get_feature(self):
        return numpy.array([self.aco])

    def get_target(self):
        return numpy.array([self.exp_rate])

    def get_notes(self):
        return [(self.name,), (self.fold,), (self.pdb_id,)]

    def to_data_frame(self):
        d = {'feature':[self.feature,], 'logkf':[self.exp_rate,],
             'fold':[self.fold,], 'pdb_id':[self.pdb_id,]}
        df = pandas.DataFrame(d, index=[self.name,])
        return df


class ContactOrderCollectionTargetData(TargetData):
    """ContactOrderCollectionTargetData reads from a table that looks like this:
    N,L,aco,rco,coc1,coc2,logkf,name,fold,pdb
    3,73,12.96,0.18,0.78,17.14,5.516,a3D,a,2A3D
    7,98,35.88,0.37,2.18,203.61,-0.617,AcP,ab,1APS
    7,98,34.30,0.35,1.94,179.04,0.365,AcP_common,ab,2ACY
    .
    .
    .
    """
    def __init__(self):
        super(ContactOrderCollectionTargetData, self).__init__()
        self.feature = None
        self.exp_rates = None

    def __len__(self):
        return len(self.feature)

    def __iter__(self):
        for i in xrange(len(self.feature)):
            yield (self.feature[i], self.exp_rates[i])

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

    def load_data(self, feature, data_path="foldkin/kings/data_for_kings_fit.txt"):
        if feature in ['N', 'L', 'aco', 'rco']:
            pass
        else:
            print "Unrecognized data feature:", feature
            return
        data_file = os.path.expanduser(data_path)
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

    def get_names(self):
        return self.names

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
