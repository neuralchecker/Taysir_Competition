from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from tqdm import tqdm

from collections import OrderedDict
import joblib


class LastTokenWeightsPickleDataSetGenerator():
    def genearte_dataset(self, model, max_query_elements, path ,batch_size = 10_000):
        total_elements = 0
        generator = UniformLengthSequenceGenerator(model.alphabet, max_seq_length = 10000000).generate_all_words()
        symbols = list(model.alphabet.symbols)
        symbols.sort()
        symbols = [model.terminal_symbol] + symbols
        cache = dict()
        #pbar = tqdm(total=max_query_elements)
        while total_elements<max_query_elements:
            queries = []
            for _ in range(batch_size):
                queries.append(next(generator))    
            total_elements += batch_size                  
            results = model.get_last_token_weights_batch(queries, symbols)                         
            results_od = [OrderedDict(zip(symbols, x)) for x in results]
            results  = dict(zip(queries, results_od))          
            cache.update(results)  
            #joblib.dump(cache, path)
            #pbar.update(batch_size)