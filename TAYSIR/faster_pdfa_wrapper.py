from faster_pdfa import FasterProbabilisticDeterministicFiniteAutomaton as FasterPDFA
from typing import Union

class MlflowFasterPDFA():
    def __init__(self, faster_pdfa: FasterPDFA) -> None:
        self._predict = faster_pdfa.sequence_probability
    
    def predict(self, model_input):
        return self._predict(model_input)
    
   
