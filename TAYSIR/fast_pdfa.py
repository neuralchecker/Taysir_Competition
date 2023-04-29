import uuid
from collections import OrderedDict
import numpy as np

class FastProbabilisticDeterministicFiniteAutomaton():
    """
    Implementation of PDFA that operates only on lists where symbols are of type int.
    This class is meant to be fast, not maintainable or interoperable with other classes in the framework

    Attributes
    ----------
    transition_function: dict[string]->dict[int]->str
        Dict containing the PDFA's transition function
    probability_function: dict[string]->OrderedDict[int]->float
        Dict containing the PDFA's probability function
    initial_state: string
        Initial state of the PDFA. A key of "states"
    """
    __slots__ = ("terminal_symbol", "both_functions", "initial_state")
    def __init__(
        self,
        alphabet: set[int],
        initial_state: str,
        transition_function: dict,
        probability_function: dict,
        terminal_symbol: int,
        name: str = None
    ):
        #self.transition_function = transition_function
        #self.probability_function = probability_function
        self.terminal_symbol = terminal_symbol
        #self._name = "PDFA - " + \
        #    str(uuid.uuid4().hex) if name is None else name
        #self._alphabet = alphabet
        self.initial_state = initial_state
        self.both_functions = OrderedDict()
        for state in transition_function.keys():
            self.both_functions[state] = OrderedDict()
            for symbol in transition_function[state].keys():
                self.both_functions[state][symbol] = (transition_function[state][symbol], probability_function[state][symbol])
            self.both_functions[state][self.terminal_symbol] = (None, probability_function[state][self.terminal_symbol])

    # def next_token_probabilities(self, sequence: list[int]):
    #     remaining = sequence
    #     actual_state = self.initial_state
    #     while len(remaining) > 0:
    #         actual_state = self.transition_function[actual_state][remaining[0]]
    #         remaining = remaining[1:]
    #     return self.probability_function[actual_state]

    def sequence_probability(self, sequence: list[int]):
        # actual_state = self.initial_state
        # product = 1
        # prob_fun = self.probability_function
        # trans_fun = self.transition_function
        # for symbol in sequence:
        #     product *= prob_fun[actual_state][symbol]
        #     if symbol == self.terminal_symbol:
        #         #assert len(remaining) == 1, "Terminal symbol should be the last symbol of the sequence"
        #         return product
        #     actual_state = trans_fun[actual_state][symbol]
        # return product
        actual_state = self.initial_state
        #product = 1
        both_funs = self.both_functions.get    
        probs = []    
        for symbol in sequence:
            next_state, proba = both_funs(actual_state)[symbol]
            probs.append(proba)
            #product *= proba
            #if symbol == self.terminal_symbol:
            #    assert len(remaining) == 1, "Terminal symbol should be the last symbol of the sequence"
             #   return product
            actual_state = next_state
        return np.array(probs, dtype=np.float32).prod()
