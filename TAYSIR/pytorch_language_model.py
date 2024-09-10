from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr

from typing import List
from collections import defaultdict
import utils

import numpy as np

class PytorchLanguageModel(ProbabilisticModel):
    
    def __init__(self, alphabet, model, name = "Pytorch NN"):
        self._alphabet = alphabet
        #2 is the ammount of symbols to represent empty sequence
        self._alphabet_len = len(alphabet) + 1
        self._model = model
        self._name = name
        self._terminal_symbol = SymbolStr(str(self._alphabet_len))
        
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
        adapted_sequence = self._adapt_sequence(sequence, add_terminal=len(sequence) == 0)       
        if len(sequence)==0:
            return utils.full_next_symbols_probas(adapted_sequence, self._model)[1]
        else:
            return utils.next_symbols_probas(adapted_sequence, self._model)
    
    def _adapt_sequence(self, sequence, add_terminal = False):
        """
        Method that converts sequence to list of ints and adds the starting token to the beggining 
        and terminal token at the end depending on the variable 'add_terminal'
        """
        adapted_seq = [self._alphabet_len-1]
        for symbol in sequence.value:
            adapted_seq.append(int(symbol.value))

        if add_terminal:
            adapted_seq.append(self._alphabet_len)
        
        return adapted_seq
    
    def raw_next_symbol_probas(self, sequence: Sequence):
        result = self.process_query(sequence)        
        return result 

    def _get_symbol_index(self, symbol):
        return int(symbol.value)+1

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
        adapted_sequence = self._adapt_sequence(sequence,  add_terminal=True)
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
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        seqs_to_query = set()
        symbols = list(self.alphabet.symbols) + [self._terminal_symbol]
        for seq in sequences:
            for required_suffix in required_suffixes:
                if required_suffix not in symbols and len(required_suffix)>1:
                    seqs_to_query.add(seq+required_suffix[:-1])
                else:
                    seqs_to_query.add(seq)

        result_dict = self.raw_eval_batch(list(seqs_to_query))
        #result_dict = dict(zip(seqs_to_query, query_results))
        results = []
        for seq in sequences:
            seq_result = []
            for required_suffix in required_suffixes:
                if required_suffix not in symbols and len(required_suffix)>1:
                    seq_result.append(result_dict[seq+required_suffix[:-1]][required_suffix[-1]])
                else:
                    if required_suffix not in symbols:
                        required_suffix = SymbolStr(required_suffix.value[0].value)
                    seq_result.append(result_dict[seq][self._get_symbol_index(required_suffix)])
            results.append(seq_result)
        
        return results
    
    def raw_eval_batch(self, sequences: List[Sequence]):
        if not hasattr(self, '_model'):
            raise AttributeError

        sequences_by_length = defaultdict(lambda: [])
        for seq in sequences:
            sequences_by_length[len(seq)].append(seq)            
        query_results = []
        seqs_to_query = []
        for length in sequences_by_length:            
            seqs =  sequences_by_length[length]
            adapted_sequences = list(map(lambda x: self._adapt_sequence(x), seqs))     
            adapted_sequences_np = np.asarray(adapted_sequences)

            #if length == 1:
            #    adapted_sequences_np = adapted_sequences_np.reshape((-1, 1, len(adapted_sequences_np[0]))) 
            if length == 0:                
                seq = Sequence()
                #adapted_sequence = self._adapt_sequence(seqs[0], add_terminal=True)    
                #adapted_sequence_np = np.asarray(adapted_sequence)
                #result = utils.full_next_symbols_probas(adapted_sequence_np, self._model)
                #model_evaluation = [result[0]]
                result = self.process_query(seq)
                model_evaluation =[result]
            else:
                model_evaluation = utils.full_next_symbols_probas_batch(adapted_sequences_np, self._model)[:,-1]
            seqs_to_query.extend(seqs)
            query_results.extend(model_evaluation)

        result_dict = dict(zip(seqs_to_query, query_results))            
        return result_dict