import numpy
import base.model
from foldkin.rate_matrix import make_rate_matrix_from_state_ids

class State(object):
    """
    A state for a MarkovStateModel.

    Attributes
    ----------
    id : string
        A label that is used to identify this state.
    is_folded_state : bool
        Whether or not this state is the folded state.
    is_unfolded_state : bool
        Whether or not this state is the unfolded state.

    Parameters
    ----------
    id_str : string
        A label that is used to identify to this state.
    """
    def __init__(self, id_str):
        self.id = id_str
        self.is_folded_state = False
        self.is_unfolded_state = False
    def __str__(self):
        return "%s %s %s" % (self.id, self.is_folded_state,
                             self.is_unfolded_state)


class Route(object):
    """
    A route for a MarkovStateModel. Routes represent
    transitions between states in such models.

    Parameters
    ----------
    start_state_id, end_state_id : string
        The identifier strings for the states that are connected by this route.
    rate_function : callable f(t)
        A function for the rate law that governs this route. The function
        takes a float argument `t` that represents time (in case the rate
        varies with time).
    direction : string
        A descriptive string that helps classify routes, e.g. in the case of
        a folding model, `direction` is either `folding` or `unfolding` to
        denote whether a route is adding or subtracting folding units.
    """
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
    """
    A MarkovStateModel consists of states and routes.
    The routes are transitions between states.

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
    """
    def __init__(self, id_str, state_enumerator, route_mapper, parameter_set,
                 noisy=False):
        super(MarkovStateModel, self).__init__()
        self.id_str = id_str
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

        self.state_id_list = []
        for i, s in enumerate(self.states):
            if s.is_folded_state:
                self.folded_index = i
            if s.is_unfolded_state:
                self.unfolded_index = i
            if s.is_first_excited_state:
                self.first_excited_index = i
            self.state_id_list.append(s.id_str)

        if self.noisy:
            for s in self.states:
                print s
            for r in self.routes:
                print r

    def __len__(self):
        return len(self.states)

    def get_parameter(self, parameter_name):
        """
        Parameters
        ----------
        parameter_name : string
            Get the current value of parameter with name `parameter_name`.
        """
        return self.parameter_set.get_parameter(parameter_name)

    def get_id(self):
        return self.id_str

    def compute_partition_fcn(self, beta):
        """
        Computes the partition function `Q` of the model at a
        specified temperature:
        :math:`Q = \sum_i w_i(\beta)`, where :math:`w_i` is the Boltzmann
        weight of state `i` at :math:`beta`.

        Parameters
        ----------
        beta : float
            :math:`\beta = (k_b T)^{-1}`
            where :math:`k_b` is the Boltzmann constant and `T` is temperature.
        """
        boltzmann_factor_array = self.compute_boltzmann_factors(beta)
        Q = boltzmann_factor_array.sum()
        population_array = boltzmann_factor_array / Q
        return Q, population_array

    def compute_unfolded_partition_fcn(self, beta):
        """
        Computes the partition function `Q` of the model at a
        specified temperature, minus the folded state(s).
        :math:`Z = \sum_i w_i(\beta)`, where :math:`w_i` is the Boltzmann
        weight of state `i` at :math:`beta`.

        Parameters
        ----------
        beta : float
            :math:`\beta = (k_b T)^{-1}`
            where :math:`k_b` is the Boltzmann constant and `T` is temperature.
        """
        boltzmann_factor_array = self.compute_boltzmann_factors(beta)
        Q_0 = boltzmann_factor_array.sum() - boltzmann_factor_array[self.folded_index]
        return Q_0

    def build_rate_matrix(self, time):
        """
        Build a rate matrix for computing the dynamics of the model.

        Parameters
        ----------
        time : float
            Time elapsed since start of dynamics calculation. Necessary because
            some of the rates may vary with time.
        """
        rate_matrix = self._build_rate_matrix_from_routes(time)
        return rate_matrix

    def _build_rate_matrix_from_routes(self, time):
        """
        Build a rate matrix from `routes` list. An alternative would be to load
        the rate matrix from a database (but not yet implemented).

        Parameters
        ----------
        time : float
            Time elapsed since start of dynamics calculation. Necessary because
            some of the rates may vary with time.
        """
        rate_matrix = make_rate_matrix_from_state_ids(
                        index_id_list=self.state_id_list,
                        column_id_list=self.state_id_list)
        for r in self.routes:
            start_id = r.start_state
            end_id = r.end_state
            this_rate = r.rate_function(time)
            rate_matrix.set_rate(start_id, end_id, this_rate)
        rate_matrix.balance_transition_rates()
        return rate_matrix
