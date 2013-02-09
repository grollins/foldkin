import nose.tools
import numpy
from foldkin.parameter_set_distribution import ParameterSetDistribution
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet

@nose.tools.istest
def correctly_selects_optimal_parameters():
    ps_distribution = ParameterSetDistribution()
    ps = CoopModelParameterSet()
    ps.update_from_array( numpy.array([1.0, 1.0, 1.0, 1.0]) )
    ps_distribution.add_parameter_set(ps, 100.)
    ps.update_from_array( numpy.array([2.0, 2.0, 2.0, 2.0]) )
    ps_distribution.add_parameter_set(ps, 1.)
    ps.update_from_array( numpy.array([3.0, 3.0, 3.0, 3.0]) )
    ps_distribution.add_parameter_set(ps, 1000.)

    best_ps_dict = ps_distribution.get_best_parameter_set()
    nose.tools.eq_(best_ps_dict['score'], 1.0)
    nose.tools.eq_(best_ps_dict['log_K_ss'], 2.0)
    nose.tools.eq_(best_ps_dict['log_K_ter'], 2.0)
    nose.tools.eq_(best_ps_dict['log_K_f'], 2.0)
    nose.tools.eq_(best_ps_dict['log_k0'], 2.0)
