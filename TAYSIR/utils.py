import torch

def predict(sequence, model):
    """ Define the function that takes a sequence as a list of integers and return the decision of your extracted model. 
    The function does not need your model as argument. See baseline notebooks for examples."""
    
    #For instance, if you want to submit the original model:
    if hasattr(model, 'predict'): #RNN
        if len(sequence) == 0:
            return False
        value = model.predict(model.one_hot_encode(sequence)) 
        return value
    else: #Transformer
        """
        Note: In this function, add 2 to each int in the word before being input to the model,
        since ids 0 and 1 are used as special tokens.
            0 : padding id
            1 : classification token id
        Args:
            word: list of integers 
        """
        word = [ [1] + [ a+2 for a in sequence ] ]
        word = torch.IntTensor(word)
        with torch.no_grad():
            out = model(word)
            return (out.logits.argmax().item())

def full_next_symbols_probas(sequence, model):   
    return full_next_symbols_probas_batch([sequence],model)[0]   
    if not hasattr(model, 'distilbert'):
        value, hiddens = model.forward_lm(model.one_hot_encode(sequence))
        return value.detach().numpy()
    else: #Transformer
        def make_future_masks(words:torch.Tensor):
            masks = (words != 0)
            b,l = masks.size()
            #x = einops.einsum(masks, masks, "b i, b j -> b i j")
            x = torch.einsum("bi,bj->bij",masks,masks)
            x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril()
            x += torch.eye(l,dtype=torch.bool, device=x.device)
            return x.type(torch.int8)
        def predict_next_symbols(model, word):
            """
            Args:
                whole word (list): a complete sequence as a list of integers
            Returns:
                the predicted probabilities of the next ids for all prefixes (2-D ndarray)
            """
            word = [ [ a+1 for a in word ] ]
            word = torch.IntTensor(word)
            model.eval()
            with torch.no_grad():
                attention_mask = make_future_masks(word)
                out = model.forward(word, attention_mask=attention_mask)
                out = torch.nn.functional.softmax(out.logits[0], dim=1)
                return out.detach().numpy()    
        return predict_next_symbols(model, sequence)

def full_next_symbols_probas_batch(sequences, model):      
    if not hasattr(model, 'distilbert'):
        sequences = torch.stack(list(map(lambda x: model.one_hot_encode(x), sequences)))             
        value, hiddens = model.forward_lm(sequences)        
        return value.detach().numpy()
        
    else: #Transformer
        def make_future_masks(words:torch.Tensor):
            masks = (words != 0)
            b,l = masks.size()
            #x = einops.einsum(masks, masks, "b i, b j -> b i j")
            x = torch.einsum("bi,bj->bij",masks,masks)
            x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril()
            x += torch.eye(l,dtype=torch.bool, device=x.device)
            return x.type(torch.int8)
        def predict_next_symbols(model, words):
            """
            Args:
                whole word (list): a complete sequence as a list of integers
            Returns:
                the predicted probabilities of the next ids for all prefixes (2-D ndarray)
            """
            words = [ [ a+1 for a in word ] for word in words]
            words = torch.IntTensor(words)
            model.eval()
            with torch.no_grad():
                attention_mask = make_future_masks(words)
                out = model.forward(words, attention_mask=attention_mask)                
                out = torch.nn.functional.softmax(out.logits, dim=2)   
                return out.detach().numpy()  
        return predict_next_symbols(model, sequences)


def next_symbols_probas(sequence, model):     
    return full_next_symbols_probas(sequence, model)[-1]

import numpy as np
def sequence_probability(sequence, model):
    probs = full_next_symbols_probas(sequence, model)
    probas_for_word = [probs[i,a+1] for i,a in enumerate(sequence)]
    value = np.array(probas_for_word).prod()
    return float(value)
# We define a class in order to get alphabet, output alphabet and proccess query of PyTorch RNN. 
# After this we send the model to the generic LStar teacher.

from pythautomata.base_types.sequence import Sequence
from pythautomata.abstract.model import Model
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.base_types.alphabet import Alphabet

class PytorchInference(Model):
    
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
    
    @property
    def output_alphabet(self) -> Alphabet:
        return Alphabet.from_strings(['1', '0'])
    
    def process_query(self, sequence):
        adapted_sequence = self._adapt_sequence(sequence)
        return predict(adapted_sequence, self._model) == 1
    
    def _adapt_sequence(self, sequence):
        adapted_seq = [self._alphabet_len-1]
        for symbol in sequence.value:
            adapted_seq.append(int(symbol.value))
        adapted_seq.append(self._alphabet_len)
        
        return adapted_seq
    
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator

def test_model(target_model, model, max_seq_len = 100, sequence_amount = 10000, min_seq_len=50):
    sequence_generator = UniformLengthSequenceGenerator(model.alphabet, max_seq_length=max_seq_len, 
                                                        random_seed=117, min_seq_length=min_seq_len)
    sequences = sequence_generator.generate_words(sequence_amount)
    
    results = []
    for sequence in sequences:
        results.append(target_model.process_query(sequence) == model.process_query(sequence))
    
    return results

    
def test_model_w_data(target_model, model, sequences):
    results = []
    
    for sequence in enumerate(sequences):
            sequence = transform_sequence(sequence)
            results.append(model.process_query(sequence) == target_model.process_query(sequence))
            
    return results
        
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr

def transform_sequence(seq):
    #take out padding symbols
    seq = seq[1:-1]
    symbol_list = []
    for symbol in seq:
        symbol_list.append(SymbolStr(str(symbol)))
        
    return Sequence(symbol_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    