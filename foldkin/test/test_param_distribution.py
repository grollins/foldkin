import nose.tools
import numpy
from foldkin.parameter_set_distribution import ParamSetDistFactory,\
                                               ParameterSetDistribution
from foldkin.coop.coop_model_parameter_set import CoopModelParameterSet

@nose.tools.istest
def correctly_selects_optimal_parameters():
    psd_factory = ParamSetDistFactory()
    ps = CoopModelParameterSet()
    ps.update_from_array( numpy.array([1.0, 1.0, 1.0, 1.0]) )
    psd_factory.add_parameter_set(ps)
    psd_factory.add_parameter('score', 100.)
    ps.update_from_array( numpy.array([2.0, 2.0, 2.0, 2.0]) )
    psd_factory.add_parameter_set(ps)
    psd_factory.add_parameter('score', 1.)
    ps.update_from_array( numpy.array([3.0, 3.0, 3.0, 3.0]) )
    psd_factory.add_parameter_set(ps)
    psd_factory.add_parameter('score', 1000.)
    ps_distribution = psd_factory.make_psd()

    row_ind, best_series = ps_distribution.get_best_parameter_set()
    nose.tools.eq_(best_series['score'], 1.0)
    nose.tools.eq_(best_series['log_K_ss'], 2.0)
    nose.tools.eq_(best_series['log_K_ter'], 2.0)
    nose.tools.eq_(best_series['log_K_f'], 2.0)
    nose.tools.eq_(best_series['log_k0'], 2.0)
