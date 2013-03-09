import numpy
import scipy.misc
from copy import deepcopy

ALMOST_ZERO = 1e-50
ALMOST_INF = 1e300

boltz_k = 0.002 # kcal/mol/Kelvin

def n_choose_k(n,k):
    assert n > 0, "%d %d" % (n, k)
    return scipy.misc.comb(n, k)

def convert_beta_to_T(beta):
    T = 1./(beta * boltz_k)
    return T

def convert_T_to_beta(T):
    beta = 1./(boltz_k * T)
    return beta

def change_lnx_to_log10x(ln_x):
    return numpy.log10(numpy.exp(ln_x))

def change_log10x_to_lnx(log10_x):
    return numpy.log(10**log10_x)

def compute_dy_at_x(x_value, x_label, parameter_set, model_factory, y_fcn):
    lower_x_bound = x_value - (x_value*0.01)
    upper_x_bound = x_value + (x_value*0.01)

    parameter_set.set_parameter(x_label, lower_x_bound)
    lower_model = model_factory.create_model('', parameter_set)
    lower_y = y_fcn(lower_model)

    parameter_set.set_parameter(x_label, upper_x_bound)
    upper_model = model_factory.create_model('', parameter_set)
    upper_y = y_fcn(upper_model)

    dx = upper_x_bound - lower_x_bound
    dy = upper_y - lower_y
    dy_dx = dy / dx
    error_msg = "%.2f  %.2f" % (dy, dx)
    assert not numpy.isnan(dy_dx), error_msg
    return dy_dx

def compute_ddy_at_x(x_value, x_label, parameter_set, model_factory, y_fcn):
    lower_x_bound = x_value - (x_value*0.01)
    upper_x_bound = x_value + (x_value*0.01)
    lower_dydx = compute_dy_at_x(lower_x_bound, x_label, parameter_set,
                                 model_factory, y_fcn)
    upper_dydx = compute_dy_at_x(upper_x_bound, x_label, parameter_set,
                                 model_factory, y_fcn)
    dx = upper_x_bound - lower_x_bound
    ddydx = upper_dydx - lower_dydx
    ddy_ddx = ddydx / dx
    error_msg = "%.2f  %.2f" % (ddydx, dx)
    assert not numpy.isnan(ddy_ddx), error_msg
    return ddy_ddx

def make_copy(copy_me):
    return deepcopy(copy_me)

def make_score_fcn(model_factory, parameter_set, judge, data_predictor,
                   target_data):
    def f(current_parameter_array):
        parameter_set.update_from_array(current_parameter_array)
        current_model = model_factory.create_model(parameter_set)
        score, prediction = judge.judge_prediction(
                                current_model, data_predictor, target_data,
                                noisy=False)
        return score
    return f
