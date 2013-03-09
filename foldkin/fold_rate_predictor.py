import numpy
from base.data_predictor import DataPredictor
from foldkin.fold_rate_prediction import FoldRatePrediction,\
                                         FoldRateCollectionPrediction
from foldkin.util import ALMOST_ZERO, ALMOST_INF, change_lnx_to_log10x,\
                         compute_dy_at_x, compute_ddy_at_x
from foldkin.coop.coop_model import compute_lnQd

class FoldRatePredictor(DataPredictor):
    """docstring for FoldRatePredictor"""
    def __init__(self):
        super(FoldRatePredictor, self).__init__()
        self.prediction_factory = FoldRatePrediction

    def _generate_nan_error_msg(self, model, bf_array):
        error_msg = "fold rate evaluated to nan\n"
        for i, bf in enumerate(bf_array):
            error_msg += "%d  %.2e\n" % (i, bf)
        Q = bf_array.sum()
        inds = range(len(bf_array))
        inds.remove(model.folded_index)
        unfolded_weight = bf_array[inds].sum()
        first_excited_weight = bf_array[model.first_excited_index]
        error_msg += "unfolded: %.2e\n" % unfolded_weight
        error_msg += "1st excited state: %.2e\n" % first_excited_weight
        error_msg += "%.2e\n" % (first_excited_weight / unfolded_weight)
        ps = model.get_parameter_set()
        beta = ps.get_parameter('beta')
        error_msg += "%.2f\n" % beta
        log_k1 = ps.compute_log_k1_at_beta(beta)
        error_msg += "%.2f\n" % (log_k1)
        error_msg += "%s\n" % ps.as_array()
        return error_msg

    def predict_data(self, model, feature=None):
        return self.predict_fold_rate(model)

    def predict_fold_rate(self, model):
        ps = model.get_parameter_set()
        beta = ps.get_parameter('beta')
        log_k1 = ps.compute_log_k1_at_beta(beta)
        boltzmann_factor_array = model.compute_boltzmann_factors()
        boltzmann_factor_array[numpy.isinf(boltzmann_factor_array)] = ALMOST_INF
        boltzmann_factor_array[numpy.isnan(boltzmann_factor_array)] = ALMOST_ZERO
        Q = boltzmann_factor_array.sum()
        inds = range(len(boltzmann_factor_array))
        inds.remove(model.folded_index)
        Q_0 = boltzmann_factor_array[inds].sum()
        P1_eq = boltzmann_factor_array[model.first_excited_index]
        if P1_eq < ALMOST_ZERO:
            P1_eq = ALMOST_ZERO
        log_fold_rate = log_k1 + numpy.log10(P1_eq / Q_0)
        assert not numpy.isnan(log_fold_rate), self._generate_nan_error_msg(model, boltzmann_factor_array)
        return self.prediction_factory(log_fold_rate)

    def compute_lnQd_deriv(self, beta, ps, model_factory):
        dlnQd_dbeta = compute_dy_at_x(beta, 'beta', ps, model_factory,
                                      y_fcn=compute_lnQd)
        return dlnQd_dbeta

    def predict_deriv(self, beta, ps, model_factory):
        dlnQd_dbeta = self.compute_lnQd_deriv(beta, ps, model_factory)
        N = ps.get_parameter('N')
        first_excited = N - 1
        if first_excited < 2:
            exponent = 0
        elif first_excited == 2:
            exponent = 1
        elif first_excited == 3:
            exponent = 3
        elif first_excited == 4:
            exponent = 6
        elif first_excited == 5:
            exponent = 10
        elif first_excited == 6:
            exponent = 14
        elif first_excited > 6:
            exponent = 4 * first_excited - 10
        H_ss = ps.get_parameter('H_ss')
        H_ter = ps.get_parameter('H_ter')
        G_act = ps.get_parameter('G_act')
        dlnkf_dbeta = -(N-1)*H_ss - exponent*H_ter - G_act - dlnQd_dbeta
        dlog10kf_dbeta = change_lnx_to_log10x(dlnkf_dbeta)
        assert not numpy.isnan(dlog10kf_dbeta)
        return dlog10kf_dbeta

    def predict_second_deriv(self, beta, ps, model_factory):
        ddlnQd_ddbeta = compute_ddy_at_x(beta, 'beta', ps, model_factory,
                                         y_fcn=compute_lnQd)
        ddlog10kf_ddbeta = -change_lnx_to_log10x(ddlnQd_ddbeta)
        assert not numpy.isnan(ddlog10kf_ddbeta)
        return ddlog10kf_ddbeta

class UnfoldRatePredictor(DataPredictor):
    """docstring for UnfoldRatePredictor"""
    def __init__(self):
        super(UnfoldRatePredictor, self).__init__()
        self.prediction_factory = FoldRatePrediction

    def _generate_nan_error_msg(self, model, bf_array):
        error_msg = ""
        for i, bf in enumerate(bf_array):
            error_msg += "%d  %.2e\n" % (i, bf)
        folded_weight = bf_array[model.folded_index]
        first_excited_weight = bf_array[model.first_excited_index]
        error_msg += "folded: %.2e\n" % folded_weight
        error_msg += "1st excited state: %.2e\n" % first_excited_weight
        return error_msg

    def predict_data(self, model, feature=None):
        return self.predict_unfold_rate(model)

    def predict_unfold_rate(self, model):
        ps = model.get_parameter_set()
        beta = ps.get_parameter('beta')
        log_k1 = ps.compute_log_k1_at_beta(beta)
        boltzmann_factor_array = model.compute_boltzmann_factors()
        boltzmann_factor_array[numpy.isinf(boltzmann_factor_array)] = ALMOST_INF
        boltzmann_factor_array[numpy.isnan(boltzmann_factor_array)] = ALMOST_ZERO
        folded_weight = boltzmann_factor_array[model.folded_index]
        first_excited_weight = boltzmann_factor_array[model.first_excited_index]
        if first_excited_weight < ALMOST_ZERO:
            first_excited_weight = ALMOST_ZERO
        if folded_weight < ALMOST_ZERO:
            folded_weight = ALMOST_ZERO
        log_unfold_rate = log_k1 + numpy.log10(first_excited_weight / folded_weight)
        assert not numpy.isnan(log_unfold_rate), self._generate_nan_error_msg(model, boltzmann_factor_array)
        return self.prediction_factory(log_unfold_rate)

    def predict_deriv(self, model):
        ps = model.get_parameter_set()
        N = ps.get_parameter('N')
        first_excited = N - 1
        if first_excited < 2:
            exponent_N1 = 0
        elif first_excited == 2:
            exponent_N1 = 1
        elif first_excited == 3:
            exponent_N1 = 3
        elif first_excited == 4:
            exponent_N1 = 6
        elif first_excited == 5:
            exponent_N1 = 10
        elif first_excited == 6:
            exponent_N1 = 14
        elif first_excited > 6:
            exponent_N1 = 4 * first_excited - 10
        if N < 2:
            exponent_N = 0
        elif N == 2:
            exponent_N = 1
        elif N == 3:
            exponent_N = 3
        elif N == 4:
            exponent_N = 6
        elif N == 5:
            exponent_N = 10
        elif N == 6:
            exponent_N = 14
        elif N > 6:
            exponent_N = 4 * N - 10

        H_ss = ps.get_parameter('H_ss')
        H_ter = ps.get_parameter('H_ter')
        G_act = ps.get_parameter('G_act')
        G_f = ps.get_parameter('G_f')
        dlnku_dbeta = H_ss - (exponent_N1 - exponent_N)*H_ter + G_f - G_act
        dlog10ku_dbeta = change_lnx_to_log10x(dlnku_dbeta)
        assert not numpy.isnan(dlog10ku_dbeta)
        return dlog10ku_dbeta


class FoldRateCollectionPredictor(DataPredictor):
    """docstring for FoldRateCollectionPredictor"""
    def __init__(self, element_predictor):
        super(FoldRateCollectionPredictor, self).__init__()
        self.element_predictor = element_predictor()
        self.prediction_factory = FoldRateCollectionPrediction

    def predict_data(self, model_collection):
        prediction_collection = self.prediction_factory()
        for this_element in model_collection:
            element_prediction = self.element_predictor.predict_data(this_element)
            prediction_collection.add_prediction(this_element.get_id(),
                                                 element_prediction)
        return prediction_collection

