from foldkin.base.model_factory import ModelFactory
from foldkin.base.model import Model
from foldkin.zam_protein import create_zam_protein_from_pdb_id

class ContactOrderModelFactory(ModelFactory):
    """docstring for ContactOrderModelFactory"""
    def __init__(self):
        super(ContactOrderModelFactory, self).__init__()

    def create_model(self, pdb_id, parameter_set):
        new_model = ContactOrderModel(pdb_id, parameter_set)
        return new_model


class ContactOrderModel(Model):
    """docstring for ContactOrderModel"""
    def __init__(self, pdb_id, parameter_set):
        super(ContactOrderModel, self).__init__()
        self.zam_protein = create_zam_protein_from_pdb_id(pdb_id)
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def get_contact_list(self):
        contact_list = self.zam_protein.get_contact_list()
        one_letter_sequence = self.zam_protein.get_sequence()
        new_contact_list = []
        for c in contact_list:
            residue1_number = c[0]
            residue2_number = c[1]
            residue1_name = one_letter_sequence[residue1_number]
            residue2_name = one_letter_sequence[residue2_number]
            new_contact = Contact(residue1_name, residue2_name, residue1_number,
                                  residue2_number)
            new_contact_list.append(new_contact)
        return new_contact_list

class Contact(object):
    """docstring for Contact"""
    def __init__(self, residue1_name, residue2_name, residue1_number,
                 residue2_number):
        super(Contact, self).__init__()
        self.residue1_name = residue1_name
        self.residue2_name = residue2_name
        self.residue1_number = residue1_number
        self.residue2_number = residue2_number

    def get_sequence_separation(self):
        return self.residue2_number - self.residue1_number

    def get_residue_names_as_letters(self):
        return [self.residue1_name, self.residue2_name]
