import torch
import mlflow
from utils import predict, PytorchInference
import sys
import pandas as pd
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import numpy as np
from wrapper import MlflowDFA
from submit_tools_fix import save_function
from pymodelextractor.teachers.pac_comparison_strategy import PACComparisonStrategy
from pymodelextractor.teachers.general_teacher import GeneralTeacher
from pymodelextractor.factories.lstar_factory import LStarFactory
from utils import test_model_w_data

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

TRACK = 1 #always for his track
dataset_amount = 11
tested_results = []
max_extraction_time = 300

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
    target_model = PytorchInference(alphabet, model, name)
    comparator = PACComparisonStrategy(target_model_alphabet = alphabet, epsilon = 0.01, delta = 0.01)
    teacher = GeneralTeacher(target_model, comparator)
    learner = LStarFactory.get_dfa_lstar_learner(max_time=max_extraction_time)
    
    res = learner.learn(teacher)
    
    # Only run if you have time.
    #result = test_model_w_data(target_model, res.model, sequences)
    #tested_results.append(np.mean(result))
    
    mlflow_dfa = MlflowDFA(res.model)
    save_function(mlflow_dfa, len(res.model.alphabet), target_model.name)
    
    print("DATASET: " + str(DATASET) + " learned with " + str(res.info['equivalence_queries_count']) + 
          " equivalence queries and " + str(res.info['membership_queries_count']) + 
          " membership queries with a duration of " + str(res.info['duration']) + "s")
    
    