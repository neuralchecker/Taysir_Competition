from fast_pdfa import FastProbabilisticDeterministicFiniteAutomaton as FastPDFA
from typing import Union

class MlflowFastPDFA():
    def __init__(self, fast_pdfa: FastPDFA) -> None:
        self._predict = fast_pdfa.sequence_probability
    
    def predict(self, model_input):
        return self._predict(model_input)
    
   
