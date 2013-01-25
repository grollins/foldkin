import nose.tools
import mock
import numpy
from foldkin.kings.contact_order_predictor import ContactOrderCollectionPredictor,\
                                                  UniformWeightAcoPredictor
from foldkin.kings.contact_order_collection import ContactOrderCollectionFactory
from foldkin.kings.contact_order_target_data import ContactOrderCollectionTargetData

EPSILON = 0.01

@nose.tools.istest
def UniformWeightPredictionMatchesTypicalACOPrediction():
    mock_parameter_set = mock.Mock()
    logk0 = 6.0
    gamma = 0.15
    def mock_get_parameter(param):
        if param == 'logk0':
            return logk0
        elif param == 'gamma':
            return gamma
        else:
            print "Unknown parameter", param
    mock_parameter_set.get_parameter = mock_get_parameter

    # pdb_id_list = ['1TIT', '1HMK', '1SHG', '2HQI', '1FEX']
    target_data = ContactOrderCollectionTargetData()
    target_data.load_data('aco')
    target_array = target_data.get_target()
    pdb_id_list = target_data.get_pdb_ids()
    coc_factory = ContactOrderCollectionFactory(pdb_id_list)
    coc_model = coc_factory.create_model(mock_parameter_set)

    predictor = ContactOrderCollectionPredictor(UniformWeightAcoPredictor)
    prediction_collection = predictor.predict_data(coc_model)
    predicted_logkf = prediction_collection.as_array()

    aco_list = []
    for element in coc_model:
        aco_list.append( element.zam_protein.compute_aco() )
    aco_logkf = logk0 - gamma * numpy.array(aco_list)
    logkf_diff = numpy.abs(predicted_logkf - aco_logkf)
    mismatch_list = numpy.where(logkf_diff > EPSILON)[0]

    for i,p in enumerate(prediction_collection):
        print p, aco_logkf[i], target_array[i]

    for i,t in enumerate(zip(target_data.get_names(), target_array)):
        n = t[0]
        logkf = t[1]
        print n, logkf, pdb_id_list[i]

    error_msg = ''
    nose.tools.eq_(len(mismatch_list), 0, error_msg)