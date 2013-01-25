import scipy.misc

boltz_k = 0.002 # kcal/mol/Kelvin

def n_choose_k(n,k):
    assert n > 0, "%d %d" % (n, k)
    return scipy.misc.comb(n, k)

