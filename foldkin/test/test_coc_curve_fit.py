import nose.tools
from foldkin.bootstrap_selector import BootstrapSelector
from foldkin.file_archiver import FileArchiver
from foldkin.parameter_set_distribution import ParameterSetDistribution
from foldkin.scipy_optimizer import ScipyOptimizer
from foldkin.kings.coc_curve_fit import KingsFeatureModelFactory,\
                                        KingsFeatureParameterSet,\
                                        KingsFeatureJudge,\
                                        KingsFeatureDataPredictor,\
                                        KingsTargetData

@nose.tools.istest
class FitKingsModelToCollectionOfFoldRates(object):
    def make_score_fcn(self, model_factory, parameter_set,
                       judge, data_predictor, target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score, prediction = judge.judge_prediction(current_model,
                                                       data_predictor,
                                                       target_data)
            print score, parameter_set
            return score
        return f

    @nose.tools.istest
    def predict_rates(self):
        '''This test fits a collection of folding rates.
           The fitting function is y = -a * aco + a^2 * coc1 + b*coc2 + c,
           where abc are fit parameters.
        '''
        model_factory = KingsFeatureModelFactory()
        initial_parameters = KingsFeatureParameterSet()
        initial_parameters.set_parameter('a', 0.1)
        initial_parameters.set_parameter('b', 0.5)
        initial_parameters.set_parameter('c', 5.0)
        judge = KingsFeatureJudge()
        data_predictor = KingsFeatureDataPredictor()
        target_data = KingsTargetData()
        target_data.load_data()
        bs_selector = BootstrapSelector()
        optimizer = ScipyOptimizer(epsilon=1e-3)
        param_dist = ParameterSetDistribution()

        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor, target_data)
        results = optimizer.optimize_parameters(score_fcn, initial_parameters)
        new_params, score, num_iterations = results
        print new_params

        # compute prediction from optimized params
        optimized_model = model_factory.create_model(new_params)
        score, prediction = judge.judge_prediction(optimized_model, 
                                                   data_predictor,
                                                   target_data)
        archiver = FileArchiver()
        archiver.save_results(target_data, prediction,
                              "kings_fit_results.txt")
        # param_dist.save_to_file("parameter_distribution_from_curve_fit.pkl")
