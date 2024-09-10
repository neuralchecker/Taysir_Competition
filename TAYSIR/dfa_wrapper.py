from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr, Symbol
from typing import Union

class MlflowDFA():
    def __init__(self, dfa: DeterministicFiniteAutomaton) -> None:
        self.dfa = dfa

    def create_sequence(self, input_data: Union['Sequence', list[Symbol], tuple[Symbol], list[str], tuple[str], str, int, list[int], tuple[int]]) -> Sequence:
    
        # Convert input_data to a list of Symbol objects if it's a list or tuple of strings
        if isinstance(input_data, (list, tuple)):
            if all(isinstance(item, str) for item in input_data):
                input_data = [SymbolStr(item) for item in input_data]
            elif all(isinstance(item, Symbol) for item in input_data):
                input_data = list(input_data)
            elif all(isinstance(item, int) for item in input_data):
                input_data = [SymbolStr(str(item)) for item in input_data]

        # Convert input_data to a list of tuples of Symbol objects if it's a single string or integer
        elif isinstance(input_data, (str, int)):
            input_data = [SymbolStr(str(char)) for char in str(input_data)]

        # Create a Sequence object from the converted input_data
        return Sequence(input_data)
    
    def predict(self, model_input):
        sequence = self.create_sequence(model_input)

        return self.dfa.accepts(sequence)
    
   
