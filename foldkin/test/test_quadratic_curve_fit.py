import nose.tools
from foldkin.scipy_optimizer import ScipyOptimizer
import foldkin.quadratic_curve.quadratic_model as quad
from foldkin.fold_rate_judge import TemperatureDependenceJudge
from foldkin.fold_rate_target_data import TemperatureDependenceTargetData
from foldkin.file_archiver import TemperatureDependenceFileArchiver
from foldkin.util import convert_T_to_beta

@nose.tools.istest
class TestFitOneTemperatureDependence(object):
    def make_score_fcn(self, fold_model_factory, unfold_model_factory,
                       parameter_set, judge, fold_data_predictor,
                       unfold_data_predictor, fold_target_data,
                       unfold_target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_fold_model = fold_model_factory.create_model(parameter_set)
            current_unfold_model = unfold_model_factory.create_model(parameter_set)
            results = judge.judge_prediction(current_fold_model,
                                             current_unfold_model,
                                             fold_data_predictor,
                                             unfold_data_predictor,
                                             fold_target_data,
                                             unfold_target_data,
                                             noisy=True)
            score, fold_prediction, unfold_prediction = results
            print score, current_parameter_array
            return score
        return f

    @nose.tools.istest
    def predicted_rate_similar_to_true_rate(self):
        '''This example fits coop model to experimental
           rate of one protein as a function of temperature.
        '''
        fold_target_data = TemperatureDependenceTargetData()
        fold_target_data.load_data('Pin_WW', 'fold')
        unfold_target_data = TemperatureDependenceTargetData()
        unfold_target_data.load_data('Pin_WW', 'unfold')
        fold_beta_array = fold_target_data.get_feature()
        unfold_beta_array = unfold_target_data.get_feature()
        fold_id_list = range(len(fold_beta_array))
        fold_model_factory = quad.QuadFitModelFactory(fold_id_list,
                                                      fold_beta_array)
        unfold_id_list = range(len(unfold_beta_array))
        unfold_model_factory = quad.QuadFitModelFactory(unfold_id_list,
                                                        unfold_beta_array)
        initial_parameters = quad.QuadFitParameterSet()
        initial_parameters.set_parameter('x', convert_T_to_beta(300.))
        judge = TemperatureDependenceJudge()
        fold_data_predictor = quad.QuadFitDataPredictor()
        unfold_data_predictor = quad.QuadFitDataPredictor()
        score_fcn = self.make_score_fcn(fold_model_factory, unfold_model_factory,
                                        initial_parameters, judge,
                                        fold_data_predictor, unfold_data_predictor,
                                        fold_target_data, unfold_target_data)
        optimizer = ScipyOptimizer(maxfun=100)
        results = optimizer.optimize_parameters(score_fcn, initial_parameters)
        new_params, score, num_iterations = results
        print score, new_params

        current_fold_collection = fold_model_factory.create_model(new_params)
        current_unfold_collection = unfold_model_factory.create_model(new_params)
        results = judge.judge_prediction(current_fold_collection,
                                         current_unfold_collection,
                                         fold_data_predictor,
                                         unfold_data_predictor,
                                         fold_target_data,
                                         unfold_target_data)
        score, fold_prediction, unfold_prediction = results

        archiver = TemperatureDependenceFileArchiver()
        archiver.save_results(fold_target_data, fold_prediction,
                              unfold_target_data, unfold_prediction)
