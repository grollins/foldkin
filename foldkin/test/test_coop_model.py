import nose.tools
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet


@nose.tools.istest
def ModelHasCorrectNumberOfStatesAndRoutes():
    parameter_set = CoopModelParameterSet()
    model_factory = CoopModelFactory()
    model = model_factory.create_model('', parameter_set)
    num_states = model.get_num_states()
    N = parameter_set.get_parameter('N')
    expected_num_states = N + 1
    error_message = "Got model with %d states, " \
                     "expected model with %d states." % \
                     (num_states, expected_num_states)
    nose.tools.eq_(num_states, expected_num_states,
                   error_message)
    num_routes = model.get_num_routes()
    expected_num_routes = 2 * (expected_num_states - 1)
    error_message = "Got model with %d routes, " \
                    "expected model with %d routes." % \
                     (num_routes, expected_num_routes)
    nose.tools.eq_(num_routes, expected_num_routes,
                   error_message)

@nose.tools.istest
def ModelParametersAreAccessible():
    parameter_set = CoopModelParameterSet()
    model_factory = CoopModelFactory()
    model = model_factory.create_model('', parameter_set)
    N_from_model = model.get_parameter('N')
    true_N = parameter_set.get_parameter('N')
    nose.tools.eq_(N_from_model, true_N,
                   "Expected %d, got %d" % (true_N, N_from_model))
