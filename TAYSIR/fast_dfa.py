
import uuid

class FastDeterministicFiniteAutomaton():
    """
    Implementation of DFA that operates only on lists where symbols are of type int.
    This class is meant to be faster than DFA, not maintainable or interoperable with other classes in the framework
    Attributes
    ----------
    transition_function: dict[string]->dict[int]->str
        Dict containing the DFA's transition function
    initial_state: string
        Initial state of the DFA. A key of "states"
    terminal_states: set[string]
        Set containing terminal states of DFA
       
    """

    def __init__(
        self,
        alphabet: set[int],
        initial_state: str,
        transition_function: dict,
        terminal_states: set[str],
        name: str = None
    ):
        self._transition_function = transition_function
        self._name = "DFA - " + \
            str(uuid.uuid4().hex) if name is None else name
        self.initial_state = initial_state
        self._terminal_states = terminal_states
        
        
    def predict(self, input_data):
        input_data = input_data[1:-1]
        actual_state = self.initial_state
        for symbol in input_data:
            actual_state = self._transition_function[(actual_state, symbol)]
        return actual_state in self._terminal_states
    
    
        