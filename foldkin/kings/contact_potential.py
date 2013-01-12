import numpy
from thomas_dill20 import THOMAS_DILL20_DICT

class ThomasDill20(object):
    """docstring for ThomasDill20"""
    def __init__(self):
        super(ThomasDill20, self).__init__()
        self.potential_dict = THOMAS_DILL20_DICT

    def compute_energy_of_contact(self, contact):
        residue_names = contact.get_residue_names_as_letters()
        return self.potential_dict[residue_names[0]][residue_names[1]]
