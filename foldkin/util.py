import scipy.misc

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
