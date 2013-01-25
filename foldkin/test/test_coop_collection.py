import numpy
import nose.tools
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet
from foldkin.coop.coop_collection import CoopCollectionFactory
from foldkin.fold_rate_predictor import FoldRateCollectionPredictor,\
                                        FoldRatePredictor

epsilon = 0.1

@nose.tools.istest
def ModelHasCorrectNumberOfStatesAndRoutes():
    parameter_set = CoopModelParameterSet()
    parameter_array = numpy.array([-2.0, 0.3, 6.0, 5.6, 3])
    parameter_set.update_from_array(parameter_array)
    N_range = range(1,31)
    model_factory = CoopCollectionFactory(N_range, 'N', N_range)
    model_collection = model_factory.create_model(parameter_set)
    for this_model in model_collection:
        num_states = this_model.get_num_states()
        N = this_model.get_parameter('N')
        expected_num_states = N + 1
        error_message = "Got model with %d states, " \
                         "expected model with %d states." % \
                         (num_states, expected_num_states)
        nose.tools.eq_(num_states, expected_num_states,
                       error_message)
        num_routes = this_model.get_num_routes()
        expected_num_routes = 2 * (expected_num_states - 1)
        error_message = "Got model with %d routes, " \
                        "expected model with %d routes." % \
                         (num_routes, expected_num_routes)
        nose.tools.eq_(num_routes, expected_num_routes,
                       error_message)

@nose.tools.istest
def ModelPredictsSameRatesAsPreviousCode():
    previous_logkf_dict = {1:5.59999, 2:3.89242, 3:2.56388, 4:1.68419, 5:1.27634,
                           6:1.35059, 7:1.41237, 8:1.46482, 9:1.50982, 10:1.54835,
                           11:1.58049, 12:1.60527, 13:1.61999, 14:1.61933,
                           15:1.59451, 16:1.53379, 17:1.42627, 18:1.26816,
                           19:1.06556, 20:8.30434e-1, 21:5.74536e-1,
                           22:3.06393e-1, 23:3.12577e-2, 24:-2.4789e-1,
                           25:-5.2946e-1,26:-8.1257e-1, 27:-1.0967, 28:-1.3817,
                           29:-1.6674, 30:-1.9536}
    parameter_set = CoopModelParameterSet()
    parameter_array = numpy.array([-2.0, 0.5, 7.0, 5.6, 3])
    parameter_set.update_from_array(parameter_array)
    N_range = range(1,31)
    model_factory = CoopCollectionFactory(N_range, 'N', N_range)
    model_collection = model_factory.create_model(parameter_set)
    N_array = numpy.array(N_range)
    data_predictor = FoldRateCollectionPredictor(FoldRatePredictor)
    prediction = data_predictor.predict_data(model_collection)
    for id_str, this_logkf in prediction:
        logkf_from_previous_code = previous_logkf_dict[id_str]
        error_message = "Expected %.2e, got %.2e for %d" % (logkf_from_previous_code,
                                                            this_logkf, id_str)
        delta_logkf = this_logkf - logkf_from_previous_code
        nose.tools.ok_(abs(delta_logkf) < epsilon, error_message)
