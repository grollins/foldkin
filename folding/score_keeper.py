import abc
from score_function import FoldingRateScoreFunction

class ScoreKeeper(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_score(self, predicted_data_collection):
        return

class FoldingScoreKeeper(ScoreKeeper):
    """docstring for ScoreKeeper"""
    def __init__(self, true_data_collection, data_set_weights):
        super(ScoreKeeper, self).__init__()
        self.true_data_collection = true_data_collection
        self.data_set_weights = data_set_weights
        self.score_function_dict = {'kf':FoldingRateScoreFunction(),
                                    'arrhenius':None}

    def compute_score(self, predicted_data_collection):
        data_sets_and_weights = zip(predicted_data_collection,
                                    self.data_set_weights)
        total_score = 0.0
        for this_predicted_data_set, this_weight in data_sets_and_weights:
            pred_id = this_predicted_data_set.get_id()
            this_true_data_set = self.true_data_collection.get_data_set(pred_id)
            this_score_fcn = self.score_function_dict[pred_id]
            this_score = this_score_fcn.compute_score(this_true_data_set,
                                                      this_predicted_data_set)
            this_score *= this_weight
            total_score += this_score
        return total_score


class SimpleScoreKeeper(ScoreKeeper):
    def __init__(self):
        self.score_function = SimpleScoreFunction()
    def compute_score(self, predicted_data_collection):
        for predicted_data in predicted_data_collection:
            return self.score_function.compute_score(None, predicted_data)
