import nose.tools
from folding.scipy_optimizer import ScipyOptimizer
import folding.curve_fit_one_feature_model as curve
from folding.fold_rate_target_data import FoldRateCollectionTargetData
from folding.file_archiver import FileArchiver
from folding.bootstrap_selector import BootstrapSelector
from folding.parameter_set_distribution import ParameterSetDistribution

EPSILON = 0.1

@nose.tools.istest
class TestFitCurveToCollectionOfFoldRates(object):
    def make_score_fcn(self, model_factory, parameter_set,
                       judge, data_predictor, target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score, prediction = judge.judge_prediction(current_model,
                                                       data_predictor,
                                                       target_data)
            return score
        return f

    @nose.tools.istest
    def rates_are_predicted(self):
        '''This example fits a collection of folding rates
           as a function of one feature. The fitting function
           is y = a * x^b + c, where x is the feature and
           abc are fit parameters.
        '''
        model_factory = curve.CurveFitOneFeatureModelFactory()
        initial_parameters = curve.CurveFitOneFeatureParameterSet()
        initial_parameters.set_parameter('a', 0.0)
        initial_parameters.set_parameter('b', 1.0)
        initial_parameters.set_parameter('c', 0.0)
        initial_parameters.set_parameter_bounds('b', 0.1, 1.0)
        judge = curve.CurveFitOneFeatureJudge()
        data_predictor = curve.CurveFitOneFeatureDataPredictor()
        target_data = FoldRateCollectionTargetData()
        target_data.load_data('aco')
        bs_selector = BootstrapSelector()
        optimizer = ScipyOptimizer()
        param_dist = ParameterSetDistribution()

        # bootstrapped data
        for i in xrange(10):
            resampled_target_data = bs_selector.select_data(target_data)
            score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                            judge, data_predictor,
                                            resampled_target_data)
            new_params, score = optimizer.optimize_parameters(score_fcn,
                                                              initial_parameters)
            param_dist.add_parameter_set(new_params, score)

        # all data
        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor, target_data)
        new_params, score = optimizer.optimize_parameters(score_fcn,
                                                          initial_parameters)

        # compute prediction from optimized params
        optimized_model = model_factory.create_model(new_params)
        score, prediction = judge.judge_prediction(optimized_model, 
                                                   data_predictor,
                                                   resampled_target_data)
        archiver = FileArchiver()
        archiver.save_results(target_data, prediction,
                              "test/output/test_fit_curve_results.txt")

        param_dist.save_to_file("test/output/parameter_distribution_from_curve_fit.pkl")
        reloaded_param_dist = ParameterSetDistribution()
        reloaded_param_dist.load_from_file("test/output/parameter_distribution_from_curve_fit.pkl")
        nose.tools.eq_(param_dist, reloaded_param_dist,
                       "Reloaded parameter distribution doesn't match.")
