import numpy
import base.model

class State(object):
    def __init__(self, id_str):
        self.id = id_str
        self.is_folded_state = False
        self.is_unfolded_state = False
    def __str__(self):
        return "%s %s %s" % (self.id, self.is_folded_state,
                             self.is_unfolded_state)

class Route(object):
    def __init__(self, start_state, end_state, rate_function, direction):
        self.start_state = start_state
        self.end_state = end_state
        self.rate_function = rate_function
        self.direction = direction
    def __str__(self):
        my_str = "%s %s %.3e" % (self.start_state, self.end_state,
                                 self.rate_function(0.))
        return my_str

class MarkovStateModel(base.model.Model):
    def __init__(self, state_enumerator, route_mapper, parameter_set,
                 noisy=False):
        super(MarkovStateModel, self).__init__()
        self.states = state_enumerator()
        if route_mapper is None:
            self.routes = None
        else:
            self.routes = route_mapper(self.states)

        self.state_index_dict = {}
        for i, s in enumerate(self.states):
            self.state_index_dict[s.id] = i

        self.noisy = noisy
        self.parameter_set = parameter_set

        for i, s in enumerate(self.states):
            if s.is_folded_state:
                self.folded_index = i
            if s.is_unfolded_state:
                self.unfolded_index = i
            if s.is_first_excited_state:
                self.first_excited_index = i

        if self.noisy:
            for s in self.states:
                print s
            for r in self.routes:
                print r

    def __len__(self):
        return len(self.states)

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

    def compute_partition_fcn(self, beta):
        boltzmann_factor_array = self.compute_boltzmann_factors(beta)
        Q = boltzmann_factor_array.sum()
        population_array = boltzmann_factor_array / Q
        return Q, population_array

    def compute_unfolded_partition_fcn(self, beta):
        boltzmann_factor_array = self.compute_boltzmann_factors(beta)
        Q_0 = boltzmann_factor_array.sum() - boltzmann_factor_array[self.folded_index]
        return Q_0

    def rate_matrix_fcn_factory(self):
        def compute_rate_matrix(t):
            rate_matrix = self.build_rate_matrix(self.state_index_dict, 0.)
            return rate_matrix
        return compute_rate_matrix

    def build_rate_matrix(self, state_index_dict, time=0.):
        N = len(self)
        rate_matrix = numpy.zeros([N, N])
        for r in self.routes:
            start_index = state_index_dict[r.start_state]
            end_index = state_index_dict[r.end_state]
            # K[start,end] = k(start-->end)
            rate_matrix[start_index,end_index] = r.rate_function(time)
        for i in range(N):
            rate_matrix[i,i] = -numpy.sum(rate_matrix[i,:])
        return rate_matrix
