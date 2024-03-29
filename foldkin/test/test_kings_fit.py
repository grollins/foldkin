import nose.tools
from foldkin.bootstrap_selector import BootstrapSelector
from foldkin.file_archiver import FileArchiver
from foldkin.parameter_set_distribution import ParameterSetDistribution
from foldkin.scipy_optimizer import ScipyOptimizer
from foldkin.fold_rate_judge import FoldRateJudge
from foldkin.kings.contact_order_collection import ContactOrderCollectionFactory
from foldkin.kings.contact_order_parameter_set import ContactOrderParameterSet
from foldkin.kings.contact_order_target_data import ContactOrderCollectionTargetData
from foldkin.kings.contact_order_predictor import ContactOrderCollectionPredictor,\
                                                  UniformWeightAcoPredictor,\
                                                  FavorLowEnergyAcoPredictor


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
            # print score
            return score
        return f

    @nose.tools.istest
    def predict_rates(self):
        '''This test fits a collection of folding rates.
           The fitting function is y = -a * aco + a^2 * coc1 + b*coc2 + c,
           where abc are fit parameters.
        '''
        initial_parameters = ContactOrderParameterSet()
        judge = FoldRateJudge()
        data_predictor = ContactOrderCollectionPredictor(UniformWeightAcoPredictor)
        target_data = ContactOrderCollectionTargetData()
        target_data.load_data('aco')
        bs_selector = BootstrapSelector()
        resampled_target_data = bs_selector.select_data(target_data, size=10)
        pdb_id_list = list(resampled_target_data.get_pdb_ids())
        model_factory = ContactOrderCollectionFactory(pdb_id_list)
        max_iterations = 1
        optimizer = ScipyOptimizer(epsilon=1e-3, maxfun=max_iterations)
        param_dist = ParameterSetDistribution()

        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor,
                                        resampled_target_data)
        results = optimizer.optimize_parameters(score_fcn, initial_parameters)
        new_params, score, num_iterations = results
        error_msg = "Expected %d iterations, got %d" % (max_iterations, num_iterations)
        nose.tools.eq_( (num_iterations - max_iterations), 1, error_msg)
        print error_msg
        print new_params

        # compute prediction from optimized params
        optimized_model = model_factory.create_model(new_params)
        score, prediction = judge.judge_prediction(optimized_model, 
                                                   data_predictor,
                                                   resampled_target_data)
        archiver = FileArchiver()
        archiver.save_results(resampled_target_data, prediction,
                              "kings_fit_results.txt")
        # param_dist.save_to_file("parameter_distribution_from_curve_fit.pkl")
