import torch
import mlflow
from utils import predict, PytorchInference
import sys
import pandas as pd
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import numpy as np
from wrapper import MlflowDFA
from submit_tools_fix import save_function
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator
from pymodelextractor.teachers.pac_comparison_strategy import PACComparisonStrategy
from pymodelextractor.learners.observation_table_learners.translators.partial_dfa_translator \
    import PartialDFATranslator
from pymodelextractor.teachers.general_teacher import GeneralTeacher
from pymodelextractor.factories.lstar_factory import LStarFactory
from utils import test_model
from pythautomata.base_types.alphabet import Alphabet
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import traceback
import wandb
from fast_dfa_converter import FastDeterministicFiniteAutomatonConverter as Converter

def persist_results(dataset, learning_result, max_extraction_time):
    result = dict()
    extracted_model = learning_result.model

    result.update({ 
                'Instance': dataset,
                'Number of Extracted States': len(extracted_model.states) ,   
                'EquivalenceQuery': learning_result.info['equivalence_queries_count'], 
                'MembershipQuery': learning_result.info['membership_queries_count'], 
                'Duration': learning_result.info['duration'], 
                'TimeBound': max_extraction_time
                })

    wandb.config.update(result)
    wandb.finish()
    # SOON: history!
    
def run_instance(dataset, params):
    TRACK = 1
    DATASET = dataset
    max_extraction_time = params['max_extraction_time']
    max_sequence_len = params['max_sequence_len']
    min_sequence_len = params['min_sequence_len']
    epsilon = params['epsilon']
    delta = params['delta']
    
    model_name = f"models/1.{DATASET}.taysir.model"
    model = mlflow.pytorch.load_model(model_name)
    model.eval()

    file = f"datasets/1.{DATASET}.taysir.valid.words"

    alphabet = None
    sequences = []

    empty_sequence_len = 2
    with open(file) as f:
        a = f.readline()
        headline = a.split(' ')
        alphabet_size = int(headline[1].strip())
        alphabet = Alphabet.from_strings([str(x) for x in range(alphabet_size - empty_sequence_len)])     

    name = "Track: " + str(TRACK) + " - DataSet: " + str(DATASET)
    target_model = PytorchInference(alphabet, model, name)
    sequence_generator = UniformWordSequenceGenerator(alphabet, max_seq_length=max_sequence_len,
                                                        min_seq_length=min_sequence_len)
    comparator = PACComparisonStrategy(target_model_alphabet = alphabet, epsilon = epsilon, delta = delta, 
                                   sequence_generator = sequence_generator)
    teacher = GeneralTeacher(target_model, comparator)
    learner = LStarFactory.get_partial_dfa_lstar_learner(max_time=max_extraction_time)
    
    teacher_type = 'GeneralTeacher'
    sampling_type = 'UniformWordSequenceGenerator'
    learner_type = 'GeneralLStarLearner'
    
    params.update({
            'teacher_type': teacher_type, 
            'sampling_type': sampling_type, 
            'learner_type': learner_type
        })
    
     # Initialize wandb
    wandb.init(
        # Set the project where this run will be logged
        project="taysir_track_1",
        # Track hyperparameters and run metadata
        config=params
    )

    res = learner.learn(teacher)
    
    if len(res.model.states) < 10:
        res.model = PartialDFATranslator().translate(res.info['observation_table'], alphabet)
        print("Changed Model")

    res.model.name = "Dataset"+str(DATASET)+"-1Acc"
    res.model.export()
    
    # Persist metrics of training
    persist_results(dataset, res, max_extraction_time)

    wandb.finish()

    fast_dfa = Converter().to_fast_dfa(res.model)

    mlflow_dfa = fast_dfa
    save_function(mlflow_dfa, len(res.model.alphabet), target_model.name)

    # Should we remove it?
    print("DATASET: " + str(DATASET) + " learned with " + str(res.info['equivalence_queries_count']) + 
          " equivalence queries and " + str(res.info['membership_queries_count']) + 
          " membership queries with a duration of " + str(res.info['duration']) + "s with " + str(len(res.model.states)) + " states")
    
def run():
    params = dict()
    max_extraction_time = 60
    max_sequence_len = 80
    min_sequence_len = 10
    epsilon = 0.01
    delta = 0.01
    number_of_datasets = 1
    
    for i in range(1, number_of_datasets + 1):
        params[f"DATASET_{i}"] = {"max_extraction_time":max_extraction_time, "max_sequence_len":max_sequence_len, 
                                   "min_sequence_len":min_sequence_len, "epsilon":epsilon, "delta":delta}

    datasets_to_run = list(range(1, number_of_datasets + 1))
    for ds in datasets_to_run:
        try:
            dataset = f"DATASET_{ds}"
            run_instance(ds, params[dataset])        
        except Exception as e:
            print("Run Instance Exception")
            print(type(e))
            traceback.print_exc()
    
        
if __name__ == '__main__':
    run()  
    