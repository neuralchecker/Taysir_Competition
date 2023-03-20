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

# We define a class in order to get alphabet, output alphabet and proccess query of PyTorch RNN. 
# After this we send the model to the generic LStar teacher.

from pythautomata.base_types.sequence import Sequence
from pythautomata.abstract.model import Model
from pymodelextractor.teachers.teacher import Teacher
from pythautomata.base_types.alphabet import Alphabet

class PytorchInference(Model):
    
    def __init__(self, alphabet, model, name = "Pytorch NN"):
        self._alphabet = alphabet
        self._model = model
        self._name = name
        
    @property
    def name(self) -> Alphabet:
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
        adapted_seq = []
        for symbol in sequence.value:
            adapted_seq.append(int(symbol.value))
        
        return adapted_seq
    
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator

def test_model(target_model, model, sequence_length = 100, sequence_amount = 10000):
    sequence_generator = UniformLengthSequenceGenerator(model.alphabet, sequence_length)
    sequences = sequence_generator.generate_words(sequence_amount)
    
    results = []
    for sequence in sequences:
        results.append(target_model.process_query(sequence) == model.process_query(sequence))
    
    return results

    
def test_model_w_data(target_model, model, sequences):
    results = []
    
    for sequence in sequences:
            sequence = transform_sequence(sequence)
            results.append(target_model.process_query(sequence) == model.process_query(sequence))
            
    return results
        
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr

def transform_sequence(seq):
    symbol_list = []
    for symbol in seq:
        symbol_list.append(SymbolStr(str(symbol)))
        
    return Sequence(symbol_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    