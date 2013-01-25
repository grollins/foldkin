import nose.tools
from foldkin.scipy_optimizer import ScipyOptimizer
from foldkin.coop.coop_model import CoopModelFactory
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet
from foldkin.fold_rate_judge import FoldRateJudge
from foldkin.fold_rate_target_data import SingleFoldRateTargetData
from foldkin.fold_rate_predictor import FoldRateCollectionPredictor,\
                                        FoldRatePredictor
from foldkin.file_archiver import FileArchiver

EPSILON = 0.1

@nose.tools.istest
class TestFitOneFoldRate(object):
    def make_score_fcn(self, model_factory, parameter_set,
                       judge, data_predictor, target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model('', parameter_set)
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
        model_factory = CoopModelFactory()
        initial_parameters = CoopModelParameterSet()
        initial_parameters.set_parameter('N', 3)
        initial_parameters.set_parameter_bounds('log_k1', 5.5, 5.7)
        judge = FoldRateJudge()
        data_predictor = FoldRatePredictor()
        target_data = SingleFoldRateTargetData()
        target_data.load_data('a3D')
        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor, target_data)
        optimizer = ScipyOptimizer()
        results = optimizer.optimize_parameters(score_fcn, initial_parameters)
        new_params, score, num_iterations = results
        optimized_model = model_factory.create_model('', new_params)
        score, prediction = judge.judge_prediction(optimized_model, data_predictor,
                                                   target_data)
        true_logkf = target_data.get_target()[0]
        delta_logkf = prediction.compute_difference(true_logkf)
        print true_logkf, delta_logkf
        nose.tools.ok_(abs(delta_logkf) < EPSILON,
                       "Expected logkf = %.2f, off by %.2f" % (true_logkf, delta_logkf))

        archiver = FileArchiver()
        archiver.save_results(target_data, prediction, "test_one_markov_results.txt")
