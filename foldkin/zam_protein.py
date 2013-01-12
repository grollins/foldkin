import os.path
from zam.protein import Protein
from zam.sequence import SeqToAA1

DEFAULT_PDB_DIR = os.path.expanduser("~/Dropbox/11_28_2011/pdb")

def create_zam_protein_from_path(file_path):
    """docstring for create_zam_protein"""
    p = Protein(file_path)
    new_zam_protein = ZamProtein(p)
    return new_zam_protein

def create_zam_protein_from_pdb_id(pdb_id):
    """docstring for create_zam_protein"""
    file_path = os.path.join(DEFAULT_PDB_DIR, pdb_id + ".pdb")
    p = Protein(file_path)
    new_zam_protein = ZamProtein(p)
    return new_zam_protein


class ZamProtein(object):
    """docstring for ZamProtein"""
    def __init__(self, protein):
        super(ZamProtein, self).__init__()
        self.protein = protein

    def get_contact_list(self):
        return self.protein.ResContactList()

    def get_sequence(self):
        return SeqToAA1(self.protein.Seq)
