import nose.tools
import mock
from foldkin.kings.contact_order_predictor import SingleContactOrderPredictor
from foldkin.kings.contact_order_model import ContactOrderModel
from foldkin.zam_protein import create_zam_protein_from_pdb_id

EPSILON = 0.01

@nose.tools.istest
def CheckUniformWeightPredictions():
    for pdb_id in ['1TIT', '1HMK', '1SHG', '2HQI', '1FEX']:
        yield UniformWeightPredictionMatchesTypicalACOPrediction, pdb_id

def UniformWeightPredictionMatchesTypicalACOPrediction(pdb_id):
    zam_protein = create_zam_protein_from_pdb_id(pdb_id)
    zam_contact_list = zam_protein.get_contact_list()
    aco = zam_protein.compute_aco()
    logk0 = 6.0
    gamma = 0.15
    aco_logkf = logk0 - gamma * aco

    mock_parameter_set = mock.Mock()
    def mock_get_parameter(param):
        if param == 'logk0':
            return logk0
        elif param == 'gamma':
            return gamma
        else:
            print "Unknown parameter", param
    mock_parameter_set.get_parameter = mock_get_parameter
    model = ContactOrderModel(pdb_id, mock_parameter_set)

    predictor = SingleContactOrderPredictor()
    predicted_logkf = predictor.predict_data(model).as_array()[0]
    logkf_diff = abs(aco_logkf - predicted_logkf)
    error_msg = "Expected %.2f, got %.2f" % (aco_logkf, predicted_logkf)
    nose.tools.ok_(logkf_diff < EPSILON, error_msg)
    print error_msg
