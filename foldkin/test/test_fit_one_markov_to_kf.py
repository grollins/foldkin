import nose.tools
from foldkin.scipy_optimizer import ScipyOptimizer
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet
from foldkin.fold_rate_judge import FoldRateJudge
from foldkin.fold_rate_target_data import SingleFoldRateTargetData
from foldkin.fold_rate_predictor import FoldRateCollectionPredictor,\
                                        FoldRatePredictor
from foldkin.file_archiver import FileArchiver
from foldkin.util import make_score_fcn, convert_beta_to_T
from foldkin.stability_predictor import StabilityPredictor
from foldkin.stability_judge import StabilityJudge
from foldkin.stability_target_data import StabilityTargetData


EPSILON = 0.1

@nose.tools.istest
def predicted_rate_matches_to_true_rate():
    '''This example fits coop model to experimental
       rate of one protein.
    '''
    model_factory = CoopModelFactory()
    initial_parameters = CoopModelParameterSet()
    initial_parameters.set_parameter('N', 3)
    initial_parameters.set_parameter_bounds('log_k0', 5.5, 5.7)
    judge = FoldRateJudge()
    data_predictor = FoldRatePredictor()
    target_data = SingleFoldRateTargetData()
    target_data.load_data('a3D')
    score_fcn = make_score_fcn(model_factory, initial_parameters,
                                    judge, data_predictor, target_data)
    optimizer = ScipyOptimizer()
    results = optimizer.optimize_parameters(score_fcn, initial_parameters)
    new_params, score, num_iterations = results
    optimized_model = model_factory.create_model(new_params)
    score, prediction = judge.judge_prediction(optimized_model, data_predictor,
                                               target_data)
    true_logkf = target_data.get_target()[0]
    delta_logkf = prediction.compute_difference(true_logkf)
    print true_logkf, prediction.as_array()[0]
    nose.tools.ok_(abs(delta_logkf) < EPSILON,
                   "Expected logkf = %.2f, off by %.2f" % (true_logkf, delta_logkf))
    # archiver = FileArchiver()
    # archiver.save_results(target_data, prediction, "test_one_markov_results.txt")

@nose.tools.istest
def predicted_stability_matches_true_stability():
    model_factory = CoopModelFactory()
    params = CoopModelParameterSet()
    params.set_parameter('N', 10)
    params.set_parameter('log_K_ss', -1.6)
    params.set_parameter('log_K_ter', 0.3)
    params.set_parameter('log_K_f', 7.0)
    params.set_parameter('log_k0', 5.6)
    params.set_parameter_bounds('log_K_ss', -1.6, -1.6)
    params.set_parameter_bounds('log_K_ter', 0.3, 0.3)
    params.set_parameter_bounds('log_k0', 5.6, 5.6)
    params.set_parameter_bounds('log_K_f', 1.0, 30.0)
    target_data = StabilityTargetData()
    N_list = [params.get_parameter('N'),]
    T = convert_beta_to_T(params.get_parameter('beta'))
    target_data.load_data(N_list, T)
    model_factory = CoopModelFactory()
    judge = StabilityJudge()
    stability_predictor = StabilityPredictor()
    score_fcn = make_score_fcn(model_factory, params,
                               judge, stability_predictor, target_data,
                               noisy=True)
    optimizer = ScipyOptimizer()
    results = optimizer.optimize_parameters(score_fcn, params)
    new_params, score, num_iterations = results
    print new_params
