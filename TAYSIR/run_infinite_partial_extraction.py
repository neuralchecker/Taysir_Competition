import torch
import mlflow
import pickle
from utils import predict, PytorchInference
import numpy as np
from wrapper import MlflowDFA
from submit_tools_fix import save_function
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy
from pymodelextractor.teachers.pac_comparison_strategy import PACComparisonStrategy
from pymodelextractor.teachers.general_teacher import GeneralTeacher
from pymodelextractor.factories.lstar_factory import LStarFactory
from pythautomata.base_types.alphabet import Alphabet
from utils import test_model

TRACK = 1 #always for his track
DATASET = 1

max_extraction_time = 2 *60 #* 60 # 1 hora
max_sequence_len = 100
min_sequence_len = 10

counter = 0
observation_table = None

model_name = f"models/1.{DATASET}.taysir.model"
model = mlflow.pytorch.load_model(model_name)
model.eval()

file = f"datasets/1.{DATASET}.taysir.valid.words"

empty_sequence_len = 2
with open(file) as f:
    a = f.readline() #Skip first line (number of sequences, alphabet size)
    headline = a.split(' ')
    alphabet_size = int(headline[1].strip())
    alphabet = Alphabet.from_strings([str(x) for x in range(alphabet_size - empty_sequence_len)])

name = "Track: " + str(TRACK) + " - DataSet: " + str(DATASET) + "-  partial n° " + str(counter)
target_model = PytorchInference(alphabet, model, name)

while True:
    sequence_generator = UniformWordSequenceGenerator(alphabet, max_seq_length=max_sequence_len,
                                                        min_seq_length=min_sequence_len)
    comparator = PACComparisonStrategy(target_model_alphabet = alphabet, epsilon = 0.001, delta = 0.001,
                                   sequence_generator = sequence_generator)
    teacher = GeneralTeacher(target_model, comparator)
    #learner = LStarFactory.get_partial_dfa_lstar_learner(max_time=max_extraction_time)
    learner = LStarFactory.get_dfa_lstar_learner(max_time=max_extraction_time)
    name = "Track: " + str(TRACK) + " - DataSet: " + str(DATASET) + "-  partial n° " + str(counter)
    
    res = learner.learn(teacher, observation_table)
    
    observation_table = res.info['observation_table']
    
    with open('predicted_models/observation_table_' + str(counter) + '.pickle', 'wb') as handle:
        pickle.dump(observation_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Run only if you have time.
    max_seq_len1=1000
    min_seq_len1=100
    result_1 = test_model(target_model, res.model, max_seq_len=max_seq_len1, min_seq_len=min_seq_len1, sequence_amount=1000)
    max_seq_len2=1000
    min_seq_len2=900
    result_2 = test_model(target_model, res.model, max_seq_len=max_seq_len2, min_seq_len=min_seq_len2, sequence_amount=1000)
    
    res.model._exporting_strategies = [DfaDotExportingStrategy()]
    res.model.name = name + "_1"
    res.model.export()
    
    mlflow_dfa = MlflowDFA(res.model)
    save_function(mlflow_dfa, len(res.model.alphabet), target_model.name)

    print("DATASET: " + str(DATASET) + " - n°: " + str(counter) + " learned with " + str(res.info['equivalence_queries_count']) + 
          " equivalence queries and " + str(res.info['membership_queries_count']) + 
          " membership queries with a duration of " + str(res.info['duration']) + "s with " + str(len(res.model.states)) + " states")
    print("Testing results: ")
    print(" + Test 1 - min_seq_len=" + str(min_seq_len1) + " - max_seq_len=" + str(max_seq_len1) 
          + " - Result: " + str(np.mean(result_1)*100) + "%")
    print(" + Test 2 - min_seq_len=" + str(min_seq_len2) + " - max_seq_len=" + str(max_seq_len2) 
          + " - Result: " + str(np.mean(result_2)*100) + "%")

    counter += 1
    max_sequence_len = max_sequence_len * 2
    min_sequence_len = min_sequence_len * 2
    break