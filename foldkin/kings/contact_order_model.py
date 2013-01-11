from foldkin.base.model_factory import ModelFactory


class ContactOrderModelFactory(object):
    """docstring for ContactOrderModelFactory"""
    def __init__(self):
        super(ContactOrderModelFactory, self).__init__()

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        new_model = ContactOrderModel(self.parameter_set)
        return new_model


class ContactOrderModel(object):
    """docstring for ContactOrderModel"""
    def __init__(self, parameter_set):
        super(ContactOrderModel, self).__init__()
        self.parameter_set = parameter_set

