from foldkin.base.model_factory import ModelFactory
from foldkin.kings.contact_order_model import ContactOrderModelFactory


class ContactOrderCollectionFactory(ModelFactory):
    """docstring for ContactOrderCollectionFactory"""
    def __init__(self, pdb_id_list):
        super(ContactOrderCollectionFactory, self).__init__()
        self.element_model_factory = ContactOrderModelFactory()
        self.pdb_id_list = pdb_id_list

    def create_model(self, parameter_set):
        new_collection = ContactOrderCollection(parameter_set)
        for this_id in self.pdb_id_list:
            this_model = self.element_model_factory.create_model(this_id, parameter_set)
            new_collection.add_element(this_model)
        return new_collection


class ContactOrderCollection(object):
    """docstring for ContactOrderCollection"""
    def __init__(self, parameter_set):
        super(ContactOrderCollection, self).__init__()
        self.parameter_set = parameter_set
        self.collection = []

    def __iter__(self):
        for element in self.collection:
            yield element

    def add_element(self, element):
        self.collection.append(element)

    def get_element(self, id_string):
        found_element = None
        for element in self.collection:
            if element.get_id() == id_string:
                found_element = element
                break
        return found_element

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


