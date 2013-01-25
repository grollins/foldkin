from foldkin.base.model_factory import ModelFactory
from foldkin.base.model import Model
from foldkin.coop.coop_model import CoopModelFactory
from copy import deepcopy

def clone(ps):
    return deepcopy(ps)


class CoopCollectionFactory(ModelFactory):
    """docstring for CoopCollectionFactory"""
    def __init__(self, id_list, parameter_name, parameter_values):
        super(CoopCollectionFactory, self).__init__()
        assert parameter_name in ['N', 'beta']
        self.id_list = id_list
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.element_model_factory = CoopModelFactory()

    def create_model(self, parameter_set):
        new_collection = CoopCollection(parameter_set)
        for this_id, this_param_value in zip(self.id_list, self.parameter_values):
            parameter_set_clone = clone(parameter_set)
            parameter_set_clone.set_parameter(self.parameter_name,
                                              this_param_value)
            this_model = self.element_model_factory.create_model(this_id,
                                                            parameter_set_clone)
            new_collection.add_element(this_model)
        return new_collection


class CoopCollection(object):
    """docstring for CoopCollection"""
    def __init__(self, parameter_set):
        super(CoopCollection, self).__init__()
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
