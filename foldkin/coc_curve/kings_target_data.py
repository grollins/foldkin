import os
import pandas
import numpy
import foldkin.base.target_data
from copy import deepcopy

class KingsTargetData(foldkin.base.target_data.TargetData):
    """KingsTargetData reads from a table that looks like this:
    N,L,aco,rco,coc1,coc2,logkf,name,fold,pdb
    3,73,12.96,0.18,0.78,17.14,5.516,a3D,a,2A3D
    7,98,35.88,0.37,2.18,203.61,-0.617,AcP,ab,1APS
    7,98,34.30,0.35,1.94,179.04,0.365,AcP_common,ab,2ACY
    .
    .
    .
    """
    def __init__(self):
        super(KingsTargetData, self).__init__()
        self.aco = None
        self.coc1 = None
        self.coc2 = None
        self.exp_rates = None

    def __len__(self):
        return len(self.aco)

    def __iter__(self):
        my_iterator = zip(self.aco, self.coc1, self.coc2, self.exp_rates)
        for aco, coc1, coc2, exp_rate in my_iterator:
            yield (aco, coc1, coc2, exp_rate)

    def iter_feature(self):
        for aco, coc1, coc2, exp_rate in self:
            yield aco, coc1, coc2, exp_rate

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

    def load_data(self):
        data_file = os.path.expanduser("~/Dropbox/python/kinetic_db/simple_table.txt")
        data_table = pandas.read_csv(data_file, index_col=7, header=0)
        self.aco = numpy.array(data_table['aco'], numpy.float32)
        self.coc1 = numpy.array(data_table['coc1'], numpy.float32)
        self.coc2 = numpy.array(data_table['coc2'], numpy.float32)
        self.exp_rates = numpy.array(data_table['logkf'], numpy.float32)
        self.names = numpy.array(data_table.index.tolist(), numpy.str)
        self.folds = numpy.array(data_table['fold'], numpy.str)
        return

    def get_feature(self):
        '''
        Format:
            feature1 feature2 feature3
        0      1.0      25.3    -0.003
        1      3.0      26.7    -0.001
        2      4.0      29.4    -0.01
        '''
        my_feature_array = numpy.array( [list(self.aco), list(self.coc1),
                                         list(self.coc2)] )
        return my_feature_array.T

    def get_target(self):
        return numpy.array(self.exp_rates)

    def get_notes(self):
        return [self.names, self.folds]

    def make_copy_from_selection(self, inds):
        my_clone = deepcopy(self)
        my_clone.aco = my_clone.aco[inds]
        my_clone.coc1 = my_clone.coc1[inds]
        my_clone.coc2 = my_clone.coc2[inds]
        my_clone.exp_rates = my_clone.exp_rates[inds]
        my_clone.names = my_clone.names[inds]
        my_clone.folds = my_clone.folds[inds]
        return my_clone

    def to_data_frame(self):
        d = {'aco':self.aco, 'coc1':self.coc1, 'coc2':self.coc2,
             'logkf':self.exp_rates, 'fold':self.folds}
        df = pandas.DataFrame(d, index=self.names)
        return df
