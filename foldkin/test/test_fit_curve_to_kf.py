import nose.tools
from foldkin.scipy_optimizer import ScipyOptimizer
import foldkin.one_param_curve.curve_fit_one_feature_model as curve
from foldkin.fold_rate_target_data import FoldRateCollectionTargetData
from foldkin.file_archiver import FileArchiver
from foldkin.bootstrap_selector import BootstrapSelector
from foldkin.parameter_set_distribution import ParamSetDistFactory,\
                                               ParameterSetDistribution
from foldkin.util import make_score_fcn

EPSILON = 0.1

@nose.tools.istest
class TestFitCurveToCollectionOfFoldRates(object):
    @nose.tools.istest
    def rates_are_predicted(self):
        '''This example fits a collection of folding rates
           as a function of one feature. The fitting function
           is y = a * x^b + c, where x is the feature and
           abc are fit parameters.
        '''
        initial_parameters = curve.CurveFitOneFeatureParameterSet()
        initial_parameters.set_parameter('a', 0.0)
        initial_parameters.set_parameter('b', 1.0)
        initial_parameters.set_parameter('c', 0.0)
        initial_parameters.set_parameter_bounds('b', 0.1, 1.0)
        judge = curve.CurveFitOneFeatureJudge()
        data_predictor = curve.CurveFitOneFeatureDataPredictor()
        target_data = FoldRateCollectionTargetData()
        target_data.load_data('aco')
        pdb_id_list = target_data.get_pdb_ids()
        model_factory = curve.CurveFitOneFeatureModelFactory(pdb_id_list)
        bs_selector = BootstrapSelector()
        optimizer = ScipyOptimizer()
        psd_factory = ParamSetDistFactory()

        # bootstrapped data
        for i in xrange(10):
            resampled_target_data = bs_selector.select_data(target_data)
            score_fcn = make_score_fcn(
                            model_factory, initial_parameters,
                            judge, data_predictor, resampled_target_data)
            results = optimizer.optimize_parameters(score_fcn,
                                                    initial_parameters)
            new_params, score, num_iterations = results
            psd_factory.add_parameter_set(new_params)
            psd_factory.add_parameter('score', score)

        # all data
        score_fcn = make_score_fcn(model_factory, initial_parameters,
                                   judge, data_predictor, target_data)
        results = optimizer.optimize_parameters(score_fcn, initial_parameters)
        new_params, score, num_iterations = results

        # compute prediction from optimized params
        optimized_model = model_factory.create_model(new_params)
        score, prediction = judge.judge_prediction(optimized_model, 
                                                   data_predictor,
                                                   target_data)
        archiver = FileArchiver()
        archiver.save_results(target_data, prediction,
                              "test_fit_curve_results.txt")

        param_dist = psd_factory.make_psd()
        param_dist.save_to_file("parameter_distribution_from_curve_fit.pkl")
        reloaded_param_dist = ParameterSetDistribution()
        reloaded_param_dist.load_from_file(
                                "parameter_distribution_from_curve_fit.pkl")
        print param_dist
        print reloaded_param_dist
        nose.tools.eq_(param_dist, reloaded_param_dist,
                       "Reloaded parameter distribution doesn't match.")
