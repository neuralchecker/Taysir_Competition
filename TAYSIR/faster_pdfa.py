
import uuid
import numpy as np
from numba import njit


@njit
def _seq_prob(prob_fun, trans_fun, sequence, terminal_symbol):
    actual_state = 0
    product = 1
    for symbol in sequence:
        product *= prob_fun[actual_state][symbol]
        if symbol == terminal_symbol:
            #assert len(remaining) == 1, "Terminal symbol should be the last symbol of the sequence"
            return product
        actual_state = trans_fun[actual_state][symbol]
    return product


class FasterProbabilisticDeterministicFiniteAutomaton():
    """
    Implementation of PDFA that operates only on lists where symbols are of type int.
    This class is meant to be faster than fast PDFA, not maintainable or interoperable with other classes in the framework

    Attributes
    ----------
    transition_function: dict[string]->dict[int]->str
        Dict containing the PDFA's transition function
    probability_function: dict[string]->OrderedDict[int]->float
        Dict containing the PDFA's probability function
    initial_state: string
        Initial state of the PDFA. A key of "states"
    """

    def __init__(
        self,
        alphabet: set[int],
        initial_state: str,
        transition_function: dict,
        probability_function: dict,
        terminal_symbol: int,
        name: str = None
    ):
        self.transition_function = transition_function
        self.probability_function = probability_function
        self.terminal_symbol = terminal_symbol
        self._name = "PDFA - " + \
            str(uuid.uuid4().hex) if name is None else name
        self._alphabet = alphabet
        self.initial_state = initial_state
        self._instantiate_arrays()
        # for running the numba jit
        _seq_prob(self.probability_array, self.transition_array,
                  np.array([0, 0, 0], dtype=int), self.terminal_symbol)

    def _instantiate_arrays(self):
        state_mapping_int_to_str = {0: self.initial_state}
        state_mapping_str_to_int = {self.initial_state: 0}
        state_count = 1
        for state in self.transition_function.keys():
            if state != self.initial_state:
                state_mapping_int_to_str[state_count] = state
                state_mapping_str_to_int[state] = state_count
                state_count += 1
        transition_array = np.zeros(
            shape=(state_count, len(self._alphabet)), dtype=int)
        probability_array = np.zeros(
            shape=(state_count, len(self._alphabet)+1), dtype=float)
        for i in range(len(transition_array)):
            for j in range(len(transition_array[0])):
                transition_array[i][j] = state_mapping_str_to_int[self.transition_function[state_mapping_int_to_str[i]][j]]
                probability_array[i][j] = self.probability_function[state_mapping_int_to_str[i]][j]
            probability_array[i][len(
                self._alphabet)] = self.probability_function[state_mapping_int_to_str[i]][self.terminal_symbol]
        self.transition_array = transition_array
        self.probability_array = probability_array

    def next_token_probabilities(self, sequence: list[int]):
        remaining = sequence
        actual_state = self.initial_state
        while len(remaining) > 0:
            actual_state = self.transition_function[actual_state][remaining[0]]
            remaining = remaining[1:]
        return self.probability_function[actual_state]

    def sequence_probability(self, sequence: list[int]):
        sequence = np.array(sequence, dtype=int)
        return _seq_prob(self.probability_array, self.transition_array, sequence, self.terminal_symbol)
