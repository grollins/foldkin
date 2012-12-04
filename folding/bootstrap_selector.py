from data_selector import DataSelector
from sklearn.cross_validation import Bootstrap

class BootstrapSelector(DataSelector):
    """docstring for BootstrapSelector"""
    def __init__(self):
        super(BootstrapSelector, self).__init__()

    def sample_data(self, data_set):
        n = len(data_set)
        bs = Bootstrap(n, 1, train_size=n-1, test_size=1)
        train_index, test_index = bs.__iter__().next()
        train_index = list(train_index)
        test_index = list(test_index)
        inds = train_index + test_index
        new_data_set = data_set.sample(inds)
        return new_data_set
