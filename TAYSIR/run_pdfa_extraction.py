import mlflow
from pytorch_language_model import PytorchLanguageModel
from submit_tools_fix import save_function
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitioner
from pdfa_wrapper import MlflowPDFA
from submit_tools_fix import save_function

from utils import test_model

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

TRACK = 2 #always for his track
dataset_amount = 10
tested_results = []
max_extraction_time = 60
max_sequence_len = 1000
min_sequence_len = 500
epsilon = 0.01
delta = 0.01
max_states = 10000
max_query_length = 1000

for ds in range(dataset_amount):
    DATASET = ds + 1

    model_name = f"models/1.{DATASET}.taysir.model"
    model = mlflow.pytorch.load_model(model_name)
    model.eval()
    
    from pythautomata.base_types.alphabet import Alphabet

    file = f"datasets/1.{DATASET}.taysir.valid.words"

    alphabet = None
    sequences = []

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
            
        
    name = "Track: " + str(TRACK) + " - DataSet: " + str(DATASET)
    target_model = PytorchLanguageModel(alphabet, model, name)
    sequence_generator = UniformLengthSequenceGenerator(alphabet, max_seq_length=max_sequence_len,
                                                        min_seq_length=min_sequence_len)
    partitioner = QuantizationProbabilityPartitioner(10)
    comparator = WFAPartitionComparator(partitioner)
    
    teacher  = PACProbabilisticTeacher(target_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False)
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, max_extraction_time, generate_partial_hipothesis = False, pre_cache_queries_for_building_hipothesis = False,  check_probabilistic_hipothesis = False)

    res = learner.learn(teacher)    
    
    mlflow_dfa = MlflowPDFA(res.model)
    save_function(mlflow_dfa, len(res.model.alphabet), target_model.name)
    