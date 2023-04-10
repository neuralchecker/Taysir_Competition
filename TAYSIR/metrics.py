
import timeit
import joblib
import os, sys
from pythautomata.utilities import pdfa_metrics
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pythautomata.base_types.sequence import Sequence
import datetime
import numpy as np
from probabilistic_model_wrapper import MlflowProbabilisticModel
epsilon = Sequence()

# def compute_stats(target_model, extracted_model, tolerance, partitions, test_sequences = None, sample_size = 1000, max_seq_length = 20, seed = 42):
#     if test_sequences is None:
#         sg = SequenceGenerator(target_model.alphabet, max_seq_length, seed)
#         test_sequences = sg.generate_words(sample_size)
    
#     log_probability_error = pdfa_metrics.log_probability_error(target_model, extracted_model, test_sequences)
#     wer = pdfa_metrics.wer_avg(target_model, extracted_model, test_sequences)
#     ndcg = pdfa_metrics.ndcg_score_avg(target_model, extracted_model, test_sequences)
#     out_of_partition = pdfa_metrics.out_of_partition_elements(
#         target_model, extracted_model, test_sequences, partitions)
#     out_of_tolerance = pdfa_metrics.out_of_tolerance_elements(
#         target_model, extracted_model, test_sequences, tolerance)
#     absolute_error_avg = pdfa_metrics.absolute_error_avg(target_model, extracted_model, test_sequences)
#     mean_cross_entropy = pdfa_metrics.mean_cross_entropy(target_model, extracted_model, test_sequences)
#     return log_probability_error, wer,ndcg, out_of_partition, out_of_tolerance, absolute_error_avg, mean_cross_entropy

def is_unknown_proba(probability_vector):
    return np.any(probability_vector<0)

def taysir_error_rate_on_output_vector(target_model, extracted_model, test_sequences):
    # MSEx10⁶
    suffixes = list()
    suffixes.append(Sequence() + extracted_model.terminal_symbol)

    for symbol in extracted_model.alphabet.symbols:
        suffixes.append(Sequence((symbol,)))

    all_obs1 = target_model.last_token_probabilities_batch(
        test_sequences, suffixes)
    all_obs2 = extracted_model.last_token_probabilities_batch(
        test_sequences, suffixes)

    squared_error_sum = 0
    for i in range(len(all_obs1)):
        obs1 = np.asarray(all_obs1[i])
        obs2 = np.asarray(all_obs2[i])
        squared_error_sum += np.sum(np.square(obs1-obs2))

    return (10**6)*squared_error_sum / len(test_sequences)

def taysir_error_rate(target_model, extracted_model, test_sequences):
    # MSEx10⁶
    mlflow_dfa = MlflowProbabilisticModel(extracted_model)

    all_obs1 = [mlflow_dfa.predict(x) for x in test_sequences]
    all_obs2 = [target_model.sequence_probability(x) for x in test_sequences]

    squared_error_sum = 0
    for i in range(len(all_obs1)):
        obs1 = all_obs1[i]
        obs2 = all_obs2[i]
        squared_error_sum += np.square(obs1-obs2)

    return (10**6)*squared_error_sum / len(test_sequences)

    
def partial_accuracy(target_model, partial_model, partitioner, test_sequences):
    suffixes = list()
    suffixes.append(epsilon + partial_model.terminal_symbol)
   
    for symbol in partial_model.alphabet.symbols:
        suffixes.append(Sequence((symbol,)))

    all_obs1 = target_model.last_token_probabilities_batch(
       test_sequences, suffixes)
    all_obs2 = partial_model.last_token_probabilities_batch(
       test_sequences, suffixes)
        
    unknown_probas = 0
    correct_elements = 0
    for i in range(len(all_obs1)):
        obs1 = np.asarray(all_obs1[i])
        obs2 = np.asarray(all_obs2[i])
        if is_unknown_proba(obs2):
            unknown_probas += 1
        else:
            if partitioner.are_in_same_partition(obs1, obs2):
                correct_elements +=1
    accuracy = 0
    divisor = len(all_obs1)-unknown_probas
    if divisor > 0 : accuracy = correct_elements/divisor
    return {'Accuracy':accuracy, 'Unknown Results Percentage':unknown_probas/len(all_obs1)}



def compute_stats(target_model, extracted_model, partitioner, test_sequences = None, sample_size = 1000, max_seq_length = 20, seed = 42):
    if test_sequences is None:
        sg = UniformLengthSequenceGenerator(target_model.alphabet, max_seq_length, seed)
        test_sequences = sg.generate_words(sample_size)  
    result= dict()

    result.update(partial_accuracy(target_model, extracted_model, partitioner,test_sequences))
    result.update({"Taysir MSE": taysir_error_rate(target_model, extracted_model,test_sequences)})
    return result

def time_fun(function, *args):    
    t0 = timeit.default_timer()
    result = function(*args)
    t1 = timeit.default_timer()
    total_seconds = t1-t0
    return total_seconds, result

def load_pdfas(path):
    dirs = os.listdir( path )
    if len(dirs) == 0:
        raise Exception('No file found')
    pdfas = []
    for file in dirs:
        print(file)
        if file!='.ipynb_checkpoints':
            pdfa = joblib.load(path+file)
            pdfas.append(pdfa)
    return pdfas

def load_dfas(path):
    dirs = os.listdir( path )
    if len(dirs) == 0:
        raise Exception('No file found')
    dfas = []
    for file in dirs:
        print(file)
        dfa = joblib.load(path+file)
        dfas.append(dfa)
    return dfas

def get_path_for_result_file(experiment_result_path):
    return experiment_result_path+"/results_"+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+'.csv'