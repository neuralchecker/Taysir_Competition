import torch
import mlflow
print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
import sys
print("Your python version:", sys.version)
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitioner
#from pythautomata.model_exporters.wfa_image_exporter_with_partition_mapper import WFAImageExporterWithPartitionMapper
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pythautomata.base_types.alphabet import Alphabet

from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher

from utils import predict
from pytorch_language_model import PytorchLanguageModel
from last_token_weights_pickle_dataset_generator import LastTokenWeightsPickleDataSetGenerator
from pymodelextractor.utils.pickle_data_loader import PickleDataLoader
from fast_pdfa_converter import FastProbabilisticDeterministicFiniteAutomatonConverter as FastPDFAConverter

from fast_pdfa_wrapper import MlflowFastPDFA
from submit_tools_fix import save_function
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.sequence import Sequence

import utils
    
def run():
    dataset_amount = 10
    for ds in range(1,dataset_amount+1):
        DATASET = ds
        model_name = f"models/2.{DATASET}.taysir.model"
        model = mlflow.pytorch.load_model(model_name)
        
        print("\n")
        print("Model:", ds)
        print(model.eval())
        try:#RNN
            nb_letters = model.input_size -1
            cell_type = model.cell_type

            print("The alphabet contains", nb_letters, "symbols.")
            print("The type of the recurrent cells is", cell_type.__name__)
        except:
            nb_letters = model.distilbert.config.vocab_size
            print("The alphabet contains", nb_letters, "symbols.")
            print("The model is a transformer (DistilBertForSequenceClassification)")
    TRACK = 2 #always for this track
    DATASET = 7
    model_name = f"models/2.{DATASET}.taysir.model"
    alphabet = None
    sequences = []
    

    file = f"datasets/2.{DATASET}.taysir.valid.words"
    empty_sequence_len = 2
    with open(file) as f:
        a = f.readline()
        headline = a.split(' ')
        alphabet_size = int(headline[1].strip())
        alphabet = Alphabet.from_strings([str(x) for x in range(alphabet_size - empty_sequence_len)])

        for line in f:
            line = line.strip()
            seq = line.split(' ')
            seq = [int(i) for i in seq[1:]]
            sequences.append(seq)

    model = mlflow.pytorch.load_model(model_name)
    model.eval()
    name = "track_" + str(TRACK) + "_dataset_" + str(DATASET)
    target_model = PytorchLanguageModel(alphabet, model, name)
    #TESTTT--------
    symbols = list(target_model.alphabet.symbols)
    symbols.sort()
    symbols = [target_model.terminal_symbol] + symbols
    seq_010 = Sequence([SymbolStr('0'), SymbolStr('1'), SymbolStr('0')])
    target_last_token_weights = target_model.get_last_token_weights(seq_010, symbols)
    target_seq_proba = target_model.sequence_probability(seq_010)
    utils_seq_proba = utils.sequence_probability([0,1,0], model)
    utils_last_token_weights = utils.next_symbols_probas([0,1,0], model)
    #LastTokenWeightsPickleDataSetGenerator().genearte_dataset(target_model, 1000, "./test",100)

    epsilon = 0.1
    delta = 0.1
    max_states = 2
    max_query_length= 2
    max_secs = None
    sequence_generator = UniformLengthSequenceGenerator(alphabet, max_seq_length=100, min_seq_length=20)
    #dataloader = PickleDataLoader("./test")

    partitioner = QuantizationProbabilityPartitioner(10)
    comparator = WFAPartitionComparator(partitioner)
    teacher  = PACBatchProbabilisticTeacher(target_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False)
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, max_secs, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = False,  check_probabilistic_hipothesis = False)
    learning_result = learner.learn(teacher)  

    fast_pdfa = FastPDFAConverter().to_fast_pdfa(learning_result.model)
    mlflow_fast_pdfa = MlflowFastPDFA(fast_pdfa)    
    save_function(mlflow_fast_pdfa, len(learning_result.model.alphabet), target_model.name+"_TEST")
    # print("No cache")
    # print(learning_result.info)

    # teacher2  = PACBatchProbabilisticTeacher(target_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False, cache_from_dataloader=dataloader)
    # learner2 = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, max_secs, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = False,  check_probabilistic_hipothesis = False)
    # learning_result2 = learner2.learn(teacher2)  
    # print("With cache")
    # print(learning_result2.info)

if __name__ == '__main__':
    run()