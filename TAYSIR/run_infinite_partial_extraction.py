import torch
torch.set_num_threads(4)
import mlflow
import pickle
from utils import predict, PytorchInference
import sys
import pandas as pd
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import numpy as np
from wrapper import MlflowDFA
from submit_tools_fix import save_function
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
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

def get_alphabet(DATASET):
    file = f"datasets/1.{DATASET}.taysir.valid.words"

    alphabet = None

    empty_sequence_len = 2
    with open(file) as f:
        a = f.readline()
        headline = a.split(' ')
        alphabet_size = int(headline[1].strip())
        alphabet = Alphabet.from_strings([str(x) for x in range(alphabet_size - empty_sequence_len)])
        
    return alphabet

def test_model(target_model, model, sequence_generator, sequence_amount = 1000):
    sequences = sequence_generator.generate_words(sequence_amount)
    
    results = []
    for sequence in sequences:
        results.append(target_model.process_query(sequence) != model.process_query(sequence))
    
    return np.mean(results)

def persist_results(dataset, learning_result, max_extraction_time):
    result = dict()
    extracted_model = learning_result.model

    result.update({ 
                'Instance': 'Dataset: ' + str(dataset),
                'Number of Extracted States': len(extracted_model.states) ,   
                'EquivalenceQuery': learning_result.info['equivalence_queries_count'], 
                'MembershipQuery': learning_result.info['membership_queries_count'], 
                'Duration': learning_result.info['duration'], 
                'TimeBound': max_extraction_time,
                'HistoricModelsAmount': len(learning_result.info['history'])
                })

    wandb.config.update(result)
    wandb.finish()
    # SOON: history!
    
def run_instance(DATASET, alphabet, params, counter, past_model_states_amount, observation_table, sigma, sequence_generator):
    TRACK = 1
    max_extraction_time = params['max_extraction_time']
    max_sequence_len = params['max_sequence_len']
    min_sequence_len = params['min_sequence_len']
    epsilon = params['epsilon']
    delta = params['delta']
    
    model_name = f"models/1.{DATASET}.taysir.model"
    model = mlflow.pytorch.load_model(model_name)
    model.eval() 

    name = "DataSet: " + str(DATASET) + " - iteration: " + str(counter) + " - v"
    target_model = PytorchInference(alphabet, model, name)
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

    res = learner.learn(teacher, observation_table)

    res.model.name = "Dataset"+str(DATASET)+"-1Acc" + str(counter)
    res.model.export()
    
    observation_table = res.info['observation_table']
    
    with open('predicted_models/observation_table_' + str(counter) + '.pickle', 'wb') as handle:
        pickle.dump(observation_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Persist metrics of training
    persist_results(DATASET, res, max_extraction_time)

    wandb.finish()

    fast_dfa = Converter().to_fast_dfa(res.model)

    mlflow_dfa = fast_dfa
    save_function(mlflow_dfa, len(res.model.alphabet), target_model.name)

    print("DATASET: " + str(DATASET) + " learned with " + str(res.info['equivalence_queries_count']) + 
          " equivalence queries and " + str(res.info['membership_queries_count']) + 
          " membership queries with a duration of " + str(res.info['duration']) + "s with " + str(len(res.model.states)) + " states")
    print("min_sequence_len:", min_sequence_len)
    print("max_sequence_len:", max_sequence_len)
    
    if past_model_states_amount < len(res.model.states) + sigma:
        results = []
        for i in range(10):
            results.append(test_model(target_model, res.model, sequence_generator))

        mean_error = np.mean(results) 
        print("Model " + str(counter) + " with error: " + str(mean_error))
    else:
        mean_error = 1
    
    return len(res.model.states), observation_table, mean_error
    
def run():
    dataset_to_run = 8
    alphabet = get_alphabet(dataset_to_run)
    past_model_states_amount = 0
    counter = 0
    max_extraction_time = 3 * 60 * 60
    max_sequence_len = 1000
    min_sequence_len = 200
    epsilon = 0.1
    delta = 0.1
    sigma = 5
    observation_table = None
    sequence_generator = UniformLengthSequenceGenerator(alphabet, max_seq_length=max_sequence_len,
                                                        min_seq_length=min_sequence_len)
    
    while True:
        counter += 1

        # Choose datasets to run
        params = {"max_extraction_time":max_extraction_time, "max_sequence_len":max_sequence_len, 
                               "min_sequence_len":min_sequence_len, "epsilon":epsilon, "delta":delta}

        try:
            dataset = f"DATASET_{dataset_to_run}"
            states_amount, observation_table, mean_error = run_instance(dataset_to_run, 
                                                            alphabet, params,
                                                            counter, past_model_states_amount,
                                                            observation_table, sigma,
                                                            sequence_generator)
            if past_model_states_amount < states_amount + sigma:
                past_model_states_amount = states_amount
                
            if mean_error < 0.1 and max_sequence_len < 550:
                max_sequence_len += 50
                min_sequence_len += 50
                sequence_generator = UniformLengthSequenceGenerator(alphabet, max_seq_length=max_sequence_len,
                                                        min_seq_length=min_sequence_len)
        except Exception as e:
            print("Run Instance Exception")
            print(type(e))
            traceback.print_exc()
    
        
if __name__ == '__main__':
    run()  
    