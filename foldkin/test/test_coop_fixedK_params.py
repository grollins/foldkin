import nose.tools
import numpy
from foldkin.coop.coop_model_parameter_set import FixedK_SS_TER_TempDependenceParameterSet,\
                                                  FixedK_TER_TempDependenceParameterSet,\
                                                  randomize_parameter
from foldkin.util import change_log10x_to_lnx, convert_T_to_beta,\
                         convert_beta_to_T

EPSILON = 1e-6

def compute_S_ss(param_set):
    '''
    K = exp((-H+TS)*beta)
    lnK = -H*beta + T*beta*S
    lnK + H*beta = T * beta * S
    S = lnK/(T*beta) + H/T
    '''
    beta = param_set.get_parameter("beta")
    log_K_ss = param_set.compute_log_K_ss_at_beta(beta)
    lnK_ss = change_log10x_to_lnx(log_K_ss)
    H_ss = param_set.get_parameter('H_ss')
    T = convert_beta_to_T(beta)
    S_ss = lnK_ss/(T*beta) + H_ss/T
    return S_ss

def compute_S_ter(param_set):
    beta = param_set.get_parameter("beta")
    log_K_ter = param_set.compute_log_K_ter_at_beta(beta)
    lnK_ter = change_log10x_to_lnx(log_K_ter)
    H_ter = param_set.get_parameter('H_ter')
    T = convert_beta_to_T(beta)
    S_ter = lnK_ter/(T*beta) + H_ter/T
    return S_ter

@nose.tools.istest
def changing_H_automatically_updates_S_to_match_expected_K():
    params = FixedK_SS_TER_TempDependenceParameterSet()
    params.set_parameter('beta', convert_T_to_beta(300.))
    log_K_ss = -2.0
    log_K_ter = 0.5
    H_ss = -10.
    H_ter = -5.
    G_f = -20.
    G_act = 3.0
    log_k0 = 10.
    param_array = numpy.array([log_K_ss, log_K_ter, H_ss, H_ter, G_f,
                               G_act, log_k0])
    params.update_from_array(param_array)

    actual_S_ss = params.get_parameter('S_ss')
    actual_S_ter = params.get_parameter('S_ter')
    expected_S_ss = compute_S_ss(params)
    expected_S_ter = compute_S_ter(params)
    S_ss_diff = abs(actual_S_ss - expected_S_ss)
    S_ter_diff = abs(actual_S_ter - expected_S_ter)
    ss_error_msg = "Expected %.2e, got %.2e" % (expected_S_ss, actual_S_ss)
    ter_error_msg = "Expected %.2e, got %.2e" % (expected_S_ter, actual_S_ter)
    nose.tools.ok_(S_ss_diff < EPSILON, ss_error_msg)
    nose.tools.ok_(S_ter_diff < EPSILON, ter_error_msg)

    beta = params.get_parameter('beta')
    computed_log_K_ss = params.compute_log_K_ss_at_beta(beta)
    computed_log_K_ter = params.compute_log_K_ter_at_beta(beta)
    K_ss_diff = abs(log_K_ss - computed_log_K_ss)
    K_ter_diff = abs(log_K_ter - computed_log_K_ter)
    K_ss_error_msg = "Expected %.2e, got %.2e" % (log_K_ss, computed_log_K_ss)
    K_ter_error_msg = "Expected %.2e, got %.2e" % (log_K_ter, computed_log_K_ter)
    nose.tools.ok_(K_ss_diff < EPSILON, K_ss_error_msg)
    nose.tools.ok_(K_ter_diff < EPSILON, K_ter_error_msg)

    print ss_error_msg
    print ter_error_msg
    print K_ss_error_msg
    print K_ter_error_msg
    print params

@nose.tools.istest
def changing_H_ter_automatically_updates_S_ter_to_match_expected_K_ter():
    params = FixedK_TER_TempDependenceParameterSet()
    params.set_parameter('beta', convert_T_to_beta(300.))
    log_K_ter = 0.5
    params.set_parameter('log_K_ter', log_K_ter)
    H_ss = -10.
    S_ss = 1e-2
    H_ter = -5.
    G_f = -20.
    G_act = 3.0
    log_k0 = 10.
    param_array = numpy.array([log_K_ter, H_ss, H_ter, S_ss, G_f, G_act, log_k0])
    params.update_from_array(param_array)

    actual_S_ss = params.get_parameter('S_ss')
    actual_S_ter = params.get_parameter('S_ter')
    expected_S_ss = S_ss
    expected_S_ter = compute_S_ter(params)
    S_ss_diff = abs(actual_S_ss - expected_S_ss)
    S_ter_diff = abs(actual_S_ter - expected_S_ter)
    ss_error_msg = "Expected %.2e, got %.2e" % (expected_S_ss, actual_S_ss)
    ter_error_msg = "Expected %.2e, got %.2e" % (expected_S_ter, actual_S_ter)
    nose.tools.ok_(S_ss_diff < EPSILON, ss_error_msg)
    nose.tools.ok_(S_ter_diff < EPSILON, ter_error_msg)

    beta = params.get_parameter('beta')
    computed_log_K_ter = params.compute_log_K_ter_at_beta(beta)
    K_ter_diff = abs(log_K_ter - computed_log_K_ter)
    K_ter_error_msg = "Expected %.2e, got %.2e" % (log_K_ter, computed_log_K_ter)
    nose.tools.ok_(K_ter_diff < EPSILON, K_ter_error_msg)

    print ss_error_msg
    print ter_error_msg
    print K_ter_error_msg
    print params

@nose.tools.istest
def parameter_value_is_set_to_random_number_within_range():
    params = FixedK_TER_TempDependenceParameterSet()
    params.set_parameter('beta', convert_T_to_beta(300.))
    log_K_ter = 0.5
    H_ss = -10.
    S_ss = 1e-2
    H_ter = -5.
    G_f = -20.
    G_act = 3.0
    log_k0 = 10.
    param_array = numpy.array([log_K_ter, H_ss, H_ter, S_ss, G_f, G_act, log_k0])
    params.update_from_array(param_array)
    lower_bound = -10.0
    upper_bound = 0.0
    for i in xrange(100):
        params = randomize_parameter(params, 'H_ss', lower_bound, upper_bound)
        current_H_ss = params.get_parameter('H_ss')
        condition = (current_H_ss >= lower_bound) and\
                    (current_H_ss < upper_bound)
        error_msg = "Expected value between %.2f and %.2f,"\
                    "got %.2f" % (lower_bound, upper_bound, current_H_ss)
        nose.tools.ok_(condition, error_msg)
