from faster_pdfa import FasterProbabilisticDeterministicFiniteAutomaton as FasterPDFA
from typing import Union

class MlflowFasterPDFA():
    def __init__(self, faster_pdfa: FasterPDFA) -> None:
        self.faster_pdfa = faster_pdfa
    
    def predict(self, model_input):
        return self.faster_pdfa.sequence_probability(model_input)
    
   
