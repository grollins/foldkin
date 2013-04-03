import numpy
import nose.tools
from foldkin.stability_target_data import KG_StabilityCalculator
from foldkin.util import boltz_k

g0 = -1.2  # kcal/mol
lnz = numpy.log(7.54)
cp = -0.0148  # kcal/(mol*K)
Th = 373.15  # K
Ts = 385.15  # K
SEC_STRUCT_PER_RESIDUE = 0.0723

@nose.tools.istest
def computes_correct_stability():
    T = 300.  # K
    N_array = numpy.arange(1, 30)
    stab_calc = KG_StabilityCalculator()
    calculator_result = stab_calc.compute_stability(N_array, T)
    dG_per_residue = g0 + (boltz_k * T * lnz) \
                        + (cp * (T - Th))  \
                        - (T * cp * numpy.log(T/Ts))
    L_array = N_array / SEC_STRUCT_PER_RESIDUE
    dG_array = L_array * dG_per_residue
    print dG_array
    nose.tools.ok_(numpy.allclose(dG_array, calculator_result))
