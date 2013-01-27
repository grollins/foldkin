import nose.tools
import numpy
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.coop.coop_collection import CoopCollectionFactory
from foldkin.coop.coop_model_parameter_set import TemperatureDependenceParameterSet
from foldkin.fold_rate_predictor import FoldRateCollectionPredictor,\
                                        FoldRatePredictor, UnfoldRatePredictor
from foldkin.scipy_optimizer import ScipyOptimizer
from foldkin.fold_rate_judge import TemperatureDependenceJudge
from foldkin.fold_rate_target_data import TemperatureDependenceTargetData
from foldkin.file_archiver import TemperatureDependenceFileArchiver

@nose.tools.nottest
def convert_T_to_beta(T):
    return 1./(0.002 * T)

@nose.tools.istest
def fold_rate_varies_as_a_function_of_temperature():
    model_factory = CoopModelFactory()
    predictor = FoldRatePredictor()
    params = TemperatureDependenceParameterSet()
    params.set_parameter('N', 3)

    T_range = numpy.arange(270, 330, 10.)
    results = []
    for T in T_range:
        this_beta = convert_T_to_beta(T)
        params.set_parameter('beta', this_beta)
        this_model = model_factory.create_model('', params)
        prediction = predictor.predict_data(this_model)
        logkf = prediction.as_array()[0]
        print T, logkf
        results.append((T, logkf))
    error_msg = "Fold rate is independent of T"
    nose.tools.assert_not_equals(results[0][1], results[-1][1], error_msg)


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
                                             unfold_target_data)
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
        fold_model_factory = CoopCollectionFactory(fold_id_list, 'beta',
                                                   fold_beta_array)
        unfold_id_list = range(len(unfold_beta_array))
        unfold_model_factory = CoopCollectionFactory(unfold_id_list, 'beta',
                                                     unfold_beta_array)
        initial_parameters = TemperatureDependenceParameterSet()
        initial_parameters.set_parameter('N', 3)
        initial_parameters.set_parameter_bounds('log_k0', 4.0, 7.0)
        judge = TemperatureDependenceJudge()
        fold_data_predictor = FoldRateCollectionPredictor(FoldRatePredictor)
        unfold_data_predictor = FoldRateCollectionPredictor(UnfoldRatePredictor)
        score_fcn = self.make_score_fcn(fold_model_factory, unfold_model_factory,
                                        initial_parameters, judge,
                                        fold_data_predictor, unfold_data_predictor,
                                        fold_target_data, unfold_target_data)
        optimizer = ScipyOptimizer(maxfun=10)
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
