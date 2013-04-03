import numpy
from foldkin.base.target_data import TargetData
from foldkin.util import boltz_k

class StabilityTargetData(TargetData):
    """StabilityTargetData computes eqn 11 of Ghosh and Dill
    """
    def __init__(self):
        super(StabilityTargetData, self).__init__()
        self.N_array = None
        self.stability_array = None
        self.stability_calculator = KG_StabilityCalculator()

    def __len__(self):
        return len(self.N_array)

    def __iter__(self):
        for i in xrange(len(self.N_array)):
            yield (self.N_array[i], self.stability_array[i])

    def __str__(self):
        return str(self.to_data_frame())

    def iter_feature(self):
        for N, dG in self:
            yield N, dG

    def load_data(self, N_list, T):
        self.N_array = numpy.array(N_list)
        self.T = T
        self.stability_array = self._compute_stability_from_N_array()

    def _compute_stability_from_N_array(self):
        return self.stability_calculator.compute_stability(
                self.N_array, self.T)

    def get_feature(self):
        return self.N_array

    def get_target(self):
        return self.stability_array

    def to_data_frame(self):
        d = {'N':self.N_array, 'stability':self.stability_array}
        df = pandas.DataFrame(d, index=range(len(self)))
        return df


class KG_StabilityCalculator(object):
    """docstring for KG_StabilityCalculator"""
    def __init__(self):
        super(KG_StabilityCalculator, self).__init__()
        self.g0 = -1.2  # kcal/mol
        self.lnz = numpy.log(7.54)
        self.cp = -0.0148  # cal/(mol*K)
        self.Th = 373.15  # K
        self.Ts = 385.15  # K
        self.residues_per_ss = 1./0.0723

    def _convert_N_to_L(self, N_array):
        return N_array * self.residues_per_ss

    def _compute_dG_per_residue(self, T):
        dG_per_residue = self.g0 + (boltz_k * T * self.lnz) \
                                 + (self.cp * (T - self.Th))  \
                                 - (T * self.cp * numpy.log(T/self.Ts))
        return dG_per_residue

    def compute_stability(self, N_array, T):
        L_array = self._convert_N_to_L(N_array)
        dG_per_residue = self._compute_dG_per_residue(T)
        dG_array = L_array * dG_per_residue
        return dG_array
