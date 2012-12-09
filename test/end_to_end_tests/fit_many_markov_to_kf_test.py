import nose.tools
from folding.scipy_optimizer import ScipyOptimizer
from folding.coop_collection import CoopCollectionFactory
from folding.coop_model_parameter_set import CoopModelParameterSet
from folding.fold_rate_judge import FoldRateJudge
from folding.fold_rate_predictor import FoldRateCollectionPredictor
from folding.fold_rate_target_data import FoldRateCollectionTargetData
from folding.file_archiver import FileArchiver

@nose.tools.istest
class TestFitManyFoldRates(object):
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
    def predicted_rate_similar_to_true_rate(self):
        '''This example fits coop model to experimental
           rate of one protein.
        '''
        model_factory = CoopCollectionFactory('N', range(1,31))
        initial_parameters = CoopModelParameterSet()
        initial_parameters.set_parameter_bounds('log_k1', 5.5, 5.7)
        judge = FoldRateJudge()
        data_predictor = FoldRateCollectionPredictor()
        target_data = FoldRateCollectionTargetData()
        target_data.load_data('N')
        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor, target_data)
        optimizer = ScipyOptimizer()
        new_params, score = optimizer.optimize_parameters(score_fcn,
                                                          initial_parameters)
        optimized_model = model_factory.create_model(new_params)
        score, prediction = judge.judge_prediction(optimized_model,
                                                   data_predictor,
                                                   target_data)

        print new_params
        print score
        archiver = FileArchiver()
        archiver.save_results(target_data, prediction,
                              "test/output/test_many_markov_results.txt")
