import nose.tools
import numpy
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.coop.coop_model_parameter_set import FixedK_SS_TER_TempDependenceParameterSet
from foldkin.fold_rate_predictor import FoldRatePredictor, UnfoldRatePredictor
from foldkin.scipy_optimizer import ScipyOptimizer
from foldkin.file_archiver import TemperatureDependenceFileArchiver
from foldkin.temperature_judge import CoopFitCurveJudge
import foldkin.quadratic_curve.quadratic_model as quad

@nose.tools.istest
class TestFitOneTemperatureDependence(object):
    def make_score_fcn(self, model_factory,
                       parameter_set, judge, fold_rate_predictor,
                       unfold_rate_predictor, quad_fit_to_kf_params,
                       quad_fit_to_ku_params):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score = judge.judge_prediction(current_model, model_factory,
                                           fold_rate_predictor,
                                           unfold_rate_predictor,
                                           quad_fit_to_kf_params,
                                           quad_fit_to_ku_params)
            print score, current_parameter_array
            return score
        return f

    @nose.tools.istest
    def fits_coop_model_to_quad_fit_parameters(self):
        quad_fit_to_kf_params = quad.QuadFitParameterSet()
        # fit params for CI-2
        quad_fit_to_kf_params.set_parameter('x', 1.56)
        quad_fit_to_kf_params.set_parameter('y0', 4.73)
        quad_fit_to_kf_params.set_parameter('y1', 0.0)
        quad_fit_to_kf_params.set_parameter('y2', -30.02)

        quad_fit_to_ku_params = None
        # quad_fit_to_ku_params = quad.QuadFitParameterSet()
        # quad_fit_to_ku_params.set_parameter('x', 1.55)
        # quad_fit_to_ku_params.set_parameter('y0', 1.5)
        # quad_fit_to_ku_params.set_parameter('y1', 1.0)
        # quad_fit_to_ku_params.set_parameter('y2', 0.0)

        model_factory = CoopModelFactory()
        coop_parameters = FixedK_SS_TER_TempDependenceParameterSet()
        coop_parameters.set_parameter('N', 6)
        coop_parameters.set_parameter('H_ss', -6.2)
        coop_parameters.set_parameter('S_ss', 0.003)
        coop_parameters.set_parameter('H_ter', -0.55)
        coop_parameters.set_parameter('G_f', -7.0)
        coop_parameters.set_parameter('G_act', 5.0)
        coop_parameters.set_parameter('log_k0', 10.0)
        coop_parameters.set_parameter_bounds('log_K_ter', 0.0, 3.0)
        coop_parameters.set_parameter_bounds('H_ss', -20.0, 0.0)
        coop_parameters.set_parameter_bounds('S_ss', -1.0, 0.0)
        coop_parameters.set_parameter_bounds('H_ter', -20.0, 0.0)
        coop_parameters.set_parameter_bounds('G_f', -20.0, -1.0)
        coop_parameters.set_parameter_bounds('G_act', 0.0, 10.0)
        coop_parameters.set_parameter_bounds('log_k0', 3.0, 10.0)
        beta = quad_fit_to_kf_params.get_parameter('x')
        coop_parameters.set_parameter('beta', beta)

        judge = CoopFitCurveJudge()
        fold_rate_predictor = FoldRatePredictor()
        # unfold_rate_predictor = UnfoldRatePredictor()
        unfold_rate_predictor = None
        score_fcn = self.make_score_fcn(model_factory, coop_parameters, judge,
                                        fold_rate_predictor,
                                        unfold_rate_predictor,
                                        quad_fit_to_kf_params,
                                        quad_fit_to_ku_params)
        optimizer = ScipyOptimizer(maxfun=100)
        results = optimizer.optimize_parameters(score_fcn, coop_parameters,
                                                noisy=True)
        new_params, score, num_iterations = results
        print score, new_params, num_iterations
