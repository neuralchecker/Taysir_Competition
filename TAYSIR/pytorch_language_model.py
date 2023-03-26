from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
import utils

class PytorchLanguageModel(ProbabilisticModel):
    
    def __init__(self, alphabet, model, name = "Pytorch NN"):
        self._alphabet = alphabet
        #2 is the ammount of symbols to represent empty sequence
        self._alphabet_len = len(alphabet) + 1
        self._model = model
        self._name = name
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    
    def process_query(self, sequence):
        adapted_sequence = self._adapt_sequence(sequence)
        return utils.predict(adapted_sequence, self._model)
    
    def _adapt_sequence(self, sequence):
        adapted_seq = [self._alphabet_len-1]
        for symbol in sequence.value:
            adapted_seq.append(int(symbol.value))
        adapted_seq.append(self._alphabet_len)
        
        return adapted_seq