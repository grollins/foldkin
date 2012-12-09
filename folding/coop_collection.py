import base.model_factory
import base.model
import coop_model
from copy import deepcopy

def clone(ps):
    return deepcopy(ps)


class CoopCollectionFactory(base.model_factory.ModelFactory):
    """docstring for CoopCollectionFactory"""
    def __init__(self, parameter_name, parameter_values):
        super(CoopCollectionFactory, self).__init__()
        assert parameter_name is 'N'
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.coop_model_factory = coop_model.CoopModelFactory()

    def create_model(self, parameter_set):
        coop_collection = CoopCollection(parameter_set)
        for this_parameter_value in self.parameter_values:
            parameter_set_clone = clone(parameter_set)
            parameter_set_clone.set_parameter(self.parameter_name,
                                              this_parameter_value)
            this_coop_model = self.coop_model_factory.create_model(parameter_set_clone)
            coop_collection.add_element(this_parameter_value, this_coop_model)
        return coop_collection


class CoopCollection(base.model.Model):
    """docstring for CoopCollection"""
    def __init__(self, parameter_set):
        super(CoopCollection, self).__init__()
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
