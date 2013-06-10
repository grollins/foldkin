import numpy
import nose.tools
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet
from foldkin.dynamics_predictor import DynamicFoldRatePredictor

@nose.tools.istest
def PredictsFoldRateFromEnsembleDynamics():
    params = CoopModelParameterSet()
    params.set_parameter('log_K_ss', -1.6)
    params.set_parameter('log_K_ter', 0.3)
    params.set_parameter('log_K_f', 7.0)
    params.set_parameter('log_k0', 5.6)
    params.set_parameter('N', 2)
    model_factory = CoopModelFactory()
    model = model_factory.create_model(params)
    data_predictor = DynamicFoldRatePredictor(make_plots=True)
    prediction = data_predictor.predict_data(model)
    print prediction
