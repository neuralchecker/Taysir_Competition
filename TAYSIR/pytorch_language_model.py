from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr

import utils

import numpy as np

class PytorchLanguageModel(ProbabilisticModel):
    
    def __init__(self, alphabet, model, name = "Pytorch NN"):
        self._alphabet = alphabet
        #2 is the ammount of symbols to represent empty sequence
        self._alphabet_len = len(alphabet) + 1
        self._model = model
        self._name = name
        self._terminal_symbol = SymbolStr(self._alphabet_len-1)
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    @property
    def terminal_symbol(self):
        return self._terminal_symbol
    
    def process_query(self, sequence):
        adapted_sequence = self._adapt_sequence(sequence)
        return utils.next_symbols_probas(adapted_sequence, self._model)
    
    def _adapt_sequence(self, sequence):
        adapted_seq = [self._alphabet_len-1]
        for symbol in sequence.value:
            adapted_seq.append(int(symbol.value))
        adapted_seq.append(self._alphabet_len)
        
        return adapted_seq
    
    def raw_next_symbol_probas(self, sequence: Sequence):
        result = self.process_query(sequence)        
        return result 

    def _get_symbol_index(self, symbol):
        return int(symbol.value)

    def next_symbol_probas(self, sequence: Sequence):
        """
        Function that returns a dictionary with the probability of next symbols (not including padding_symbol)
        Quickly implemented, depends on raw_next_symbol_probas(sequence) 
        """                
        next_probas = self.raw_next_symbol_probas(sequence)

        symbols = list(self.alphabet.symbols) + [self._terminal_symbol]
        intermediate_dict = {}
        probas = np.zeros(len(symbols))
        for idx, symbol in enumerate(symbols):
            proba = next_probas[self._get_symbol_index(symbol)]
            intermediate_dict[symbol] = (proba, idx)
            probas[idx] = proba       

        dict_result = {}
        for symbol in intermediate_dict.keys():
            dict_result[symbol] = probas[intermediate_dict[symbol][1]]

        return dict_result      
    
    def last_token_probability(self, sequence: Sequence):       
        return self.next_symbol_probas(Sequence(sequence[:-1]))[sequence[-1]]

    def log_sequence_probability(self, sequence: Sequence):
        raise NotImplementedError
    
    def last_token_probabilities_batch(self, sequences: list[Sequence], required_suffixes: list[Sequence]) -> list[list[float]]:
        return self.get_last_token_weights_batch(sequences, required_suffixes)
    
    def sequence_probability(self, sequence: Sequence, debug = False):
        adapted_sequence = self._adapt_sequence(sequence)
        return utils.sequence_probability(adapted_sequence, self._model)
    
    def get_last_token_weights(self, sequence, required_suffixes):
        weights = list()
        alphabet_symbols_weights = self.next_symbol_probas(sequence)
        alphabet_symbols_weights = {Sequence() + k: alphabet_symbols_weights[k] for k in alphabet_symbols_weights.keys()}
        for suffix in required_suffixes:
            if suffix in alphabet_symbols_weights:
                weights.append(alphabet_symbols_weights[suffix])
            else:
                new_sequence = sequence + suffix
                new_prefix = Sequence(new_sequence[:-1])
                new_suffix = new_sequence[-1]
                next_symbol_weights = self.next_symbol_probas(new_prefix)
                weights.append(next_symbol_weights[new_suffix])
        return weights