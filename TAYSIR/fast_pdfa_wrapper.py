from fast_implementations.fast_pdfa import FastProbabilisticDeterministicFiniteAutomaton as FastPDFA
from typing import Union

class MlflowFastPDFA():
    def __init__(self, fast_pdfa: FastPDFA) -> None:
        self.fast_pdfa = fast_pdfa
    
    def predict(self, model_input):
        return self.fast_pdfa.sequence_probability(model_input)
    
   
