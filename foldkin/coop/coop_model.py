import numpy
import scipy.misc
from foldkin.base.model_factory import ModelFactory
from foldkin.markov_state_model import State, Route, MarkovStateModel

def n_choose_k(n,k):
    assert n > 0, "%d %d" % (n, k)
    return scipy.misc.comb(n, k)

class CoopModelFactory(ModelFactory):
    """docstring for CoopModelFactory"""
    def __init__(self):
        super(CoopModelFactory, self).__init__()

    def create_model(self, parameter_set):
        self.parameter_set = parameter_set
        state_enumerator = self.state_enumerator_factory()
        route_mapper = self.route_mapper_factory()
        new_model = CoopModel(state_enumerator, route_mapper,
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
        log_k1 = self.parameter_set.get_parameter('log_k1')
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
        log_K = self.parameter_set.get_parameter('log_K')
        log_alpha = self.parameter_set.get_parameter('log_alpha')
        log_epsilon = self.parameter_set.get_parameter('log_epsilon')
        K = 10**log_K
        alpha = 10**log_alpha
        epsilon = 10**log_epsilon
        folded_weight = epsilon

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
                    assert (f_state.C - u_state.C) > 0, "\n%s\n%s" % (str(u_state), str(f_state))
                    # are these states neighbors?
                    if self.connected_states(u_state, f_state):
                        # build a route between these states
                        # folding route
                        kf = self.kf_factory(u_state.C)
                        fold_route = Route(u_state.id, f_state.id,
                                                        kf, direction="folding")
                        route_list.append(fold_route)
                        # unfolding route
                        f_bf = f_state.compute_boltz_weight(N, K, alpha, folded_weight)
                        u_bf = u_state.compute_boltz_weight(N, K, alpha, folded_weight)
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
        self.C = C
        self.is_first_excited_state = False
    def __str__(self):
        return "%s %d %s %s %s" % (self.id, self.C, self.is_folded_state,
                                   self.is_unfolded_state, self.is_first_excited_state)

    def compute_boltz_weight(self, N, K, alpha, folded_weight):
        if self.is_unfolded_state:
            this_bf = n_choose_k(N, self.C) * K**self.C
        elif self.is_folded_state:
            this_bf = n_choose_k(N, self.C) * K**self.C * folded_weight
        else:
            this_bf = n_choose_k(N, self.C) * K**self.C

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

        this_alpha = alpha ** exponent
        this_bf *= this_alpha
        return this_bf

class CoopModel(MarkovStateModel):
    def __init__(self, state_enumerator, route_mapper, parameter_set, noisy=False):
        super(CoopModel, self).__init__(state_enumerator, route_mapper,
                                           parameter_set, noisy)

    def get_num_states(self):
        return len(self.states)

    def get_num_routes(self):
        return len(self.routes)

    def compute_boltzmann_factors(self):
        N = self.get_parameter('N')
        log_K = self.get_parameter('log_K')
        log_alpha = self.get_parameter('log_alpha')
        log_epsilon = self.get_parameter('log_epsilon')
        K = 10**log_K
        alpha = 10**log_alpha
        epsilon = 10**log_epsilon
        folded_weight = epsilon
        boltzmann_factor_list = []
        for this_state in self.states:
            this_bf = this_state.compute_boltz_weight(N, K, alpha, folded_weight)
            boltzmann_factor_list.append(this_bf)
        return numpy.array(boltzmann_factor_list)
