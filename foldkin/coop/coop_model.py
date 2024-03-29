import numpy
import pandas
from foldkin.util import n_choose_k
from foldkin.base.model_factory import ModelFactory
from foldkin.markov_state_model import State, Route, MarkovStateModel
from foldkin.util import ALMOST_INF, ALMOST_ZERO
from foldkin.probability_vector import ProbabilityVector

def compute_lnQd(coop_model):
    boltzmann_factor_array = coop_model.compute_boltzmann_factors()
    boltzmann_factor_array[numpy.isinf(boltzmann_factor_array)] = ALMOST_INF
    boltzmann_factor_array[numpy.isnan(boltzmann_factor_array)] = ALMOST_ZERO
    Q = boltzmann_factor_array.sum()
    inds = range(len(boltzmann_factor_array))
    inds.remove(coop_model.folded_index)
    Qd = boltzmann_factor_array[inds].sum()
    return numpy.log(Qd)

class CoopModelFactory(ModelFactory):
    """docstring for CoopModelFactory"""
    def __init__(self):
        super(CoopModelFactory, self).__init__()

    def create_model(self, parameter_set, id_str=''):
        self.parameter_set = parameter_set
        state_enumerator = self.state_enumerator_factory()
        route_mapper = self.route_mapper_factory()
        new_model = CoopModel(id_str, state_enumerator, route_mapper,
                              self.parameter_set, noisy=False)
        C_array = numpy.zeros( [len(new_model)] )
        for i,s in enumerate(new_model.states):
            C_array[i] = s.C
        new_model.C_array = C_array
        return new_model

    def state_enumerator_factory(self):
        N = self.parameter_set.get_parameter('N')
        C_array = numpy.arange(0, N+1, 1)
        def enumerate_states():
            state_list = []
            for i in xrange(len(C_array)):
                this_C = C_array[i]
                new_state = CoopState(str(this_C), this_C)
                if this_C == 0:
                    new_state.is_unfolded_state = True
                if this_C == N:
                    new_state.is_folded_state = True
                if this_C == N-1:
                    new_state.is_first_excited_state = True
                state_list.append(new_state)
            return state_list
        return enumerate_states

    def kf_factory(self, C):
        N = self.parameter_set.get_parameter('N')
        beta = self.parameter_set.get_parameter('beta')
        log_k1 = self.parameter_set.compute_log_k1_at_beta(beta)
        k1 = 10**log_k1
        def kf_fcn(t):
            S = N - C
            kf = S * k1
            return kf
        return kf_fcn

    def ku_factory(self, kf, f_boltz_factor, u_boltz_factor):
        def ku_fcn(t):
            ku = kf * u_boltz_factor / f_boltz_factor
            return ku
        return ku_fcn

    def connected_states(self, u_state, f_state):
        # add routes between states that differ by one
        # correct unit
        return (f_state.C - u_state.C) == 1

    def route_mapper_factory(self):
        N = self.parameter_set.get_parameter('N')
        beta = self.parameter_set.get_parameter('beta')
        log_K_ss = self.parameter_set.compute_log_K_ss_at_beta(beta)
        log_K_ter = self.parameter_set.compute_log_K_ter_at_beta(beta)
        log_K_f = self.parameter_set.compute_log_K_f_at_beta(beta)
        K_ss = 10**log_K_ss
        K_ter = 10**log_K_ter
        K_f = 10**log_K_f

        def map_routes(states):
            route_list = []
            for i, s1 in enumerate(states):
                for j, s2 in enumerate(states):
                    if j <= i:
                        # start loop where j == i+1
                        continue
                    # Which state has more correct? (bigger C)
                    # We'll make u_state the one with smaller C.
                    if s1.C >= s2.C:
                        f_state = s1
                        u_state = s2
                    elif s1.C < s2.C:
                        f_state = s2
                        u_state = s1
                    else:
                        assert False, "Logic Error"
                    error_msg = "\n%s\n%s" % (str(u_state), str(f_state))
                    assert (f_state.C - u_state.C) > 0, error_msg
                    # are these states neighbors?
                    if self.connected_states(u_state, f_state):
                        # build a route between these states
                        # folding route
                        kf = self.kf_factory(u_state.C)
                        fold_route = Route(u_state.id, f_state.id,
                                           kf, direction="folding")
                        route_list.append(fold_route)
                        # unfolding route
                        f_bf = f_state.compute_boltz_weight(N, K_ss, K_ter, K_f)
                        u_bf = u_state.compute_boltz_weight(N, K_ss, K_ter, K_f)
                        ku = self.ku_factory(kf(0.0), f_bf, u_bf)
                        unfold_route = Route(f_state.id, u_state.id,
                                             ku, direction="unfolding")
                        route_list.append(unfold_route)
                    else:
                        # don't build a route between this pair of states
                        continue
            return route_list
        return map_routes

class CoopState(State):
    def __init__(self, id_str, C):
        super(CoopState, self).__init__(id_str)
        self.id_str = id_str
        self.C = C
        self.is_first_excited_state = False
    def __str__(self):
        return "%s %d %s %s %s" % (self.id_str, self.C, self.is_folded_state,
                                   self.is_unfolded_state, self.is_first_excited_state)

    def _get_ter_exponent(self):
        if self.C < 2:
            exponent = 0
        elif self.C == 2:
            exponent = 1
        elif self.C == 3:
            exponent = 3
        elif self.C == 4:
            exponent = 6
        elif self.C == 5:
            exponent = 10
        elif self.C == 6:
            exponent = 14
        elif self.C > 6:
            exponent = 4 * self.C - 10
        return exponent

    def compute_boltz_components(self, N, K_ss, K_ter, K_f):
        multiplicity = n_choose_k(N, self.C)
        ss_factor = K_ss ** self.C
        exponent = self._get_ter_exponent()
        ter_factor = K_ter ** exponent
        if self.is_folded_state:
            folded_factor = K_f
        else:
            folded_factor = 1.0
        return multiplicity, ss_factor, ter_factor, folded_factor

    def compute_boltz_weight(self, N, K_ss, K_ter, K_f):
        components = self.compute_boltz_components(N, K_ss, K_ter, K_f)
        multiplicity, ss_factor, ter_factor, folded_factor = components
        bf = multiplicity * ss_factor * ter_factor * folded_factor
        return bf

class CoopModel(MarkovStateModel):
    """
    CoopModel is a MarkovStateModel for protein folding. The model consists
    of folding units (secondary structures) that are either folded or unfolded.

    Parameters
    ----------
    id_str : string
        A descriptor used to distinguish this model from other models.
    state_enumerator : callable f()
        Generates a list of states for the model.
    route_mapper : callable f(state_list)
        Generates a list of routes for the model.
    parameter_set : ParameterSet
    noisy : bool, optional
        Whether the model should print additional output.

    Attributes
    ----------
    states : list
    routes : list
    state_id_list : list
    state_index_dict : dict
        `state_index_dict[s]` is the integer index to the element of `states` that
        corresponds to the state with id `s`.
    folded_index, unfolded_index, first_excited_state : int
        indices to the elements of `states` that correspond to the folded,
        unfolded, and first-excited states, respectively.
    C_array : numpy ndarray
        An array of integers that span the range from 0 to N, inclusive.
    """
    def __init__(self, id_str, state_enumerator, route_mapper, parameter_set,
                 noisy=False):
        super(CoopModel, self).__init__(id_str, state_enumerator, route_mapper,
                                           parameter_set, noisy)

    def get_parameter_set(self):
        return self.parameter_set

    def get_C_array(self):
        return self.C_array

    def get_num_states(self):
        return len(self.states)

    def get_num_routes(self):
        return len(self.routes)

    def compute_boltzmann_factors(self):
        ps = self.get_parameter_set()
        N = ps.get_parameter('N')
        beta = ps.get_parameter('beta')
        log_K_ss = ps.compute_log_K_ss_at_beta(beta)
        log_K_ter = ps.compute_log_K_ter_at_beta(beta)
        log_K_f = ps.compute_log_K_f_at_beta(beta)
        K_ss = 10**log_K_ss
        K_ter = 10**log_K_ter
        K_f = 10**log_K_f
        boltzmann_factor_list = []
        for this_state in self.states:
            this_bf = this_state.compute_boltz_weight(N, K_ss, K_ter, K_f)
            boltzmann_factor_list.append(this_bf)
        return numpy.array(boltzmann_factor_list)

    def compute_boltzmann_components(self):
        ps = self.get_parameter_set()
        N = ps.get_parameter('N')
        beta = ps.get_parameter('beta')
        log_K_ss = ps.compute_log_K_ss_at_beta(beta)
        log_K_ter = ps.compute_log_K_ter_at_beta(beta)
        log_K_f = ps.compute_log_K_f_at_beta(beta)
        K_ss = 10**log_K_ss
        K_ter = 10**log_K_ter
        K_f = 10**log_K_f
        multiplicity_list = []
        ss_list = []
        ter_list = []
        folded_list = []
        boltzmann_factor_list = []
        for this_state in self.states:
            components = this_state.compute_boltz_components(
                                        N, K_ss, K_ter, K_f)
            multiplicity, ss_factor, ter_factor, folded_factor = components
            multiplicity_list.append( multiplicity )
            ss_list.append( ss_factor )
            ter_list.append( ter_factor )
            folded_list.append( folded_factor )
            this_bf = this_state.compute_boltz_weight(N, K_ss, K_ter, K_f)
            boltzmann_factor_list.append(this_bf)
        multiplicity_array = numpy.array(multiplicity_list)
        ss_array = numpy.array(ss_list)
        ter_array = numpy.array(ter_list)
        folded_array = numpy.array(folded_list)
        boltzmann_factor_array = numpy.array(boltzmann_factor_list)
        Q = boltzmann_factor_array.sum()
        component_dict = {}
        component_dict['multiplicity'] = numpy.log(multiplicity_array)
        component_dict['ss'] = numpy.log(ss_array)
        component_dict['ter'] = numpy.log(ter_array)
        component_dict['folded'] = numpy.log(folded_array)
        component_dict['total_bf'] = numpy.log(boltzmann_factor_array)
        component_dict['Q'] = numpy.log(Q)
        component_dict['G'] = component_dict['Q']\
                                - component_dict['multiplicity']\
                                - component_dict['ss']\
                                - component_dict['ter']\
                                - component_dict['folded']
        df = pandas.DataFrame(component_dict, index=self.state_id_list)
        return df

    def get_init_prob_vec(self):
        pv = ProbabilityVector()
        pv.series = pandas.Series(0.0, index=self.state_id_list)
        pv.set_state_probability(self.state_id_list[self.unfolded_index], 1.0)
        return pv
