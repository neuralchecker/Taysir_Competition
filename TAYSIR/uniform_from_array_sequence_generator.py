import random
import numpy as np
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import Symbol
from pythautomata.base_types.sequence import Sequence
from random import seed, randint

class UniformFromArraySequenceGenerator(SequenceGenerator):
    def __init__(self, alphabet: Alphabet, lenghts, random_seed: int = 21):
        self._alphabet = alphabet
        self._seed = random_seed
        seed(self._seed)
        self._length_distribution = lenghts 
       
    def generate_words(self, number_of_words: int):
        lengths = random.choices(self._length_distribution, k = number_of_words)
        result = np.empty(number_of_words, dtype=Sequence)
        for index in range(number_of_words):
            length = lengths[index]
            result[index] = self.generate_single_word(length)
        return result
    
    def generate_single_word(self, length):
        value = []
        list_symbols = list(self._alphabet.symbols)
        list_symbols.sort()
        for _ in range(length):
            position = randint(0, len(list_symbols) - 1)
            symbol = list_symbols[position]
            value.append(symbol)
        return Sequence(value)
    