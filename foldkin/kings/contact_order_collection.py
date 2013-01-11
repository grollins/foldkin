from foldkin.base.model_factory import ModelFactory
from foldkin.kings.contact_order_model import ContactOrderModelFactory


class ContactOrderCollectionFactory(ModelFactory):
    """docstring for ContactOrderCollectionFactory"""
    def __init__(self):
        super(ContactOrderCollectionFactory, self).__init__()
        self.contact_order_model_factory = ContactOrderModelFactory()

    def create_model(self, parameter_set):
        contact_order_collection = ContactOrderCollection(parameter_set)
        return contact_order_collection


class ContactOrderCollection(object):
    """docstring for ContactOrderCollection"""
    def __init__(self, parameter_set):
        super(ContactOrderCollection, self).__init__()
        self.parameter_set = parameter_set
        self.collection = {}

    def __iter__(self):
        for key, element in self.collection.iteritems():
            yield key, element

    def add_element(self, parameter_value, element):
        if self.collection.has_key(parameter_value):
            return
        else:
            self.collection[parameter_value] = element

    def get_element(self, parameter_value):
        if self.collection.has_key(parameter_value):
            return self.collection[parameter_value]
        else:
            return None

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)


