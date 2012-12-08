import os
import pandas
import numpy
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

    def get_feature(self):
        return numpy.array([self.num_ss])

    def get_target(self):
        return numpy.array([self.exp_rate])

    def get_notes(self):
        return []

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

    def load_data(self, feature):
        if feature in ['N', 'L', 'aco', 'rco']:
            pass
        else:
            print "Unrecognized data feature:", feature
            return
        data_file = os.path.expanduser("~/Dropbox/python/kinetic_db/simple_table.txt")
        data_table = pandas.read_csv(data_file, index_col=5, header=0)
        self.feature = data_table[feature]
        self.exp_rate = data_table['logkf']
        self.names = data_table.index.tolist()
        self.folds = data_table['fold']
        return

    def get_feature(self):
        return numpy.array(self.feature)

    def get_target(self):
        return numpy.array(self.exp_rate)

    def get_notes(self):
        return [self.names, self.folds]
