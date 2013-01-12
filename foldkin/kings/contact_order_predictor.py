import numpy
from foldkin.base.data_predictor import DataPredictor
from foldkin.kings.contact_potential import ThomasDill20
from foldkin.kings.contact_order_prediction import SingleContactOrderPrediction,\
                                                   ContactOrderCollectionPrediction

class SingleContactOrderPredictor(DataPredictor):
    """docstring for ContactOrderPredictor"""
    def __init__(self):
        super(SingleContactOrderPredictor, self).__init__()
        self.prediction_factory = SingleContactOrderPrediction
        self.contact_potential = ThomasDill20()

    def choose_weight(self, energy):
        if energy < -0.6:
            weight = 0.1
        elif energy >= -0.6 and energy < -0.01:
            weight = 0.5
        elif energy >= -0.01 and energy < 0.01:
            weight = 1.0
        elif energy >= 0.01:
            weight = 1.5
        else:
            print "Unexpected energy", energy
        return weight

    def predict_data(self, model):
        logk0 = model.get_parameter('logk0')
        gamma = model.get_parameter('gamma')
        contact_list = model.get_contact_list()

        first_order_aco_term = 0.0
        for c in contact_list:
            seq_separation = c.get_sequence_separation()
            this_energy = self.contact_potential.compute_energy_of_contact(c)
            this_weight = self.choose_weight(this_energy)
            first_order_aco_term += (this_weight * gamma) * abs(seq_separation)
        first_order_aco_term /= len(contact_list)

        second_order_aco_term = 0.0
        coc1_term = 0.0
        coc2_term = 0.0
        logkf = logk0 - first_order_aco_term + second_order_aco_term + coc1_term + coc2_term
        return self.prediction_factory(logkf)


class ContactOrderCollectionPredictor(DataPredictor):
    """docstring for ContactOrderCollectionPredictor"""
    def __init__(self):
        super(ContactOrderCollectionPredictor, self).__init__()
        self.element_predictor = SingleContactOrderPredictor()
        self.prediction_factory = ContactOrderCollectionPrediction

    def predict_data(self, model_collection, feature_array):
        prediction_collection = self.prediction_factory()
        for pdb_id, this_element in model_collection:
            element_prediction = self.element_predictor.predict_data(this_element)
            prediction_collection.add_prediction(element_prediction)
        return prediction_collection
