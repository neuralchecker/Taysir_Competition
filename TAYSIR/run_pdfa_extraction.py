import mlflow
from pytorch_language_model import PytorchLanguageModel
from submit_tools_fix import save_function
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from fast_pdfa_converter import FastProbabilisticDeterministicFiniteAutomatonConverter as FastPDFAConverter
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher
from pymodelextractor.teachers.sample_batch_probabilistic_teacher import SampleBatchProbabilisticTeacher
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pymodelextractor.learners.other_learners.ensemble_probabilistic_learner import EnsembleProbabilisticLearner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitioner
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.sequence import Sequence
from probabilistic_model_wrapper import MlflowProbabilisticModel
from pymodelextractor.utils.pickle_data_loader import PickleDataLoader
from fast_pdfa_wrapper import MlflowFastPDFA
from faster_pdfa_wrapper import MlflowFasterPDFA
from submit_tools_fix import save_function
import torch
import metrics
import datetime
import pandas as pd
import os
import joblib
import traceback
import wandb

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(4)

TRACK = 2 #always for this track
#max_extraction_time = 30
#max_sequence_len = 100
#min_sequence_len = 2

#max_sequence_len_transformer = 100#500 tops
#min_sequence_len_transformer = 2

#epsilon = 0.01
#delta = 0.01
#max_states = 10000
#max_query_length = 1000

def load_model(ds):
    model_name = f"models/2.{ds}.taysir.model"
    model = mlflow.pytorch.load_model(model_name)
    model.eval()
    return model

def get_alphabet_and_validation_sequences(ds):
    file = f"datasets/2.{ds}.taysir.valid.words"
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
            seq = line.split(' ')[1:]
            seq = [SymbolStr(i) for i in seq[1:]]            
            sequences.append(Sequence(seq))
    return alphabet, sequences

def create_model(alphabet, model, ds):
  name = "track_" + str(TRACK) + "_dataset_" + str(ds)
  target_model = PytorchLanguageModel(alphabet, model, name)
  return target_model

def show(result, ds):
   print("DATASET: " + str(ds) + " learned with " + str(result.info['equivalence_queries_count']) + 
          " equivalence queries and " + str(result.info['last_token_weight_queries_count']) + " membership queries "+
          " with " + str(len(result.model.weighted_states)) + " states")

def get_path_for_result_file_name(path):
    return path+"/results_"+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+'.csv'

def persist_ensemble_results(ds, learning_result, stats, path_for_results_file, path_for_framework_models, max_extraction_time):
    for model in learning_result.model._models:
        instance = "ensemble_"+str(ds)+model.name
        persist_results(instance, learning_result.info[model.name], stats, path_for_results_file, path_for_framework_models, max_extraction_time)


def persist_results(ds, learning_result, stats, path_for_results_file, path_for_framework_models, max_extraction_time, log_to_wandb):
    result = dict()
    extracted_model = learning_result.model
    if learning_result.info['observation_tree'] is None:
        tree_depth = 0
        inner_nodes = 0
    else:
        tree_depth = learning_result.info['observation_tree'].depth
        inner_nodes = len(learning_result.info['observation_tree'].inner_nodes)
    result.update({ 
                'Instance': ds,
                'Number of Extracted States': len(extracted_model.weighted_states) ,   
                'LastTokenQuery': learning_result.info['last_token_weight_queries_count'], 
                'EquivalenceQuery': learning_result.info['equivalence_queries_count'], 
                'NumberOfStatesExceeded': learning_result.info['NumberOfStatesExceeded'],
                'QueryLengthExceeded':learning_result.info['QueryLengthExceeded'], 
                'TimeExceeded': learning_result.info['TimeExceeded'],
                'Tree Depth': tree_depth,
                'Inner Nodes': inner_nodes,
                'TimeBound': max_extraction_time
                })
    result.update(stats)
    if log_to_wandb: wandb.config.update(result)
    #dfresults = pd.DataFrame([result], columns = result.keys())     
    #dfresults.to_csv(path_for_results_file, mode = 'a', header = not os.path.exists(path_for_results_file))
    name = wandb.run.name if log_to_wandb else "test" 
    joblib.dump(value=learning_result.model, filename=path_for_framework_models+"/"+str(ds)+"_"+name)
    
    if log_to_wandb: wandb.finish()


def run_instance(ds, path_for_results_file, path_for_framework_models, params, ensemble, use_cache, use_sampling_teacher, log_to_wandb):
    DATASET = ds
    model = load_model(DATASET)
    alphabet, validation_sequences = get_alphabet_and_validation_sequences(DATASET)
    target_model = create_model(alphabet, model, DATASET) 
    max_sequence_len = params['max_sequence_len']
    min_sequence_len = params['min_sequence_len']
    epsilon = params['epsilon']
    delta = params['delta']
    max_states = params['max_states']
    max_query_length = params['max_query_length']
    max_extraction_time = params['max_extraction_time']
    partitions = params['partitions']

    sampling_type = "UniformLengthSequenceGenerator"
    sequence_generator = UniformLengthSequenceGenerator(alphabet, max_seq_length=max_sequence_len, min_seq_length=min_sequence_len)  
    partitioner = QuantizationProbabilityPartitioner(partitions)
    comparator = WFAPartitionComparator(partitioner)   
    dataloader = None
    if use_cache:
        dataloader = PickleDataLoader("./data_caches/"+target_model.name) 
    if use_sampling_teacher:    
        teacher_type = "SampleBatchProbabilisticTeacher"
        teacher = SampleBatchProbabilisticTeacher(model = target_model, comparator = comparator, sequence_generator=sequence_generator, max_seq_length=max_sequence_len, full_prefix_set=True,  cache_from_dataloader=dataloader)
    else:
        teacher_type = "PACProbabilisticTeacher"
        teacher  = PACProbabilisticTeacher(target_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False, cache_from_dataloader=dataloader)
    
    if ensemble:
        learner_type = "EnsembleProbabilisticLearner"
        learners = []
        for i in range(5):
            l = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, max_extraction_time, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = False,  check_probabilistic_hipothesis = False, mean_distribution_for_partial_hipothesis=True)
            learners.append(l.learn)        
        learner = EnsembleProbabilisticLearner(learning_functions=learners)
    else:
        learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, max_extraction_time, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = False,  check_probabilistic_hipothesis = False, mean_distribution_for_partial_hipothesis=True, exhaust_counterexample = True)
        learner_type = "BoundedPDFAQuantizationNAryTreeLearner"
    
    params.update({
            'teacher_type': teacher_type, 
            'sampling_type': sampling_type, 
            'learner_type': learner_type
        })
    if log_to_wandb:
        wandb.init(
            # Set the project where this run will be logged
            project="taysir_track_2",
            # Track hyperparameters and run metadata
            config=params
        )            
    
    result = learner.learn(teacher)    
    if ensemble:
        print("Learning finished")
        mlflow_ensemble = MlflowProbabilisticModel(result.model)
        save_function(mlflow_ensemble, len(result.model.alphabet), target_model.name)
    else:
        show(result, DATASET)
        #mlflow_pdfa = MlflowPDFA(result.model)
        fast_pdfa = FastPDFAConverter().to_fast_pdfa(result.model)
        #faster_pdfa = FastPDFAConverter().to_faster_pdfa(result.model)
        mlflow_fast_pdfa = MlflowFastPDFA(fast_pdfa)
        #mlflow_faster_pdfa = MlflowFasterPDFA(faster_pdfa)
    
        #save_function(mlflow_pdfa, len(result.model.alphabet), target_model.name+"_SLOW")
        name = wandb.run.name if log_to_wandb else "test"
        save_function(mlflow_fast_pdfa, len(result.model.alphabet), target_model.name+"_" + name)
        #save_function(mlflow_faster_pdfa, len(result.model.alphabet), target_model.name+"_FASTER")

    test_sequences = sequence_generator.generate_words(1000)
    stats = metrics.compute_stats(target_model, result.model,partitioner, test_sequences, validation_sequences)
    print(stats)
    if ensemble:
        persist_ensemble_results(DATASET, result, stats, path_for_results_file, path_for_framework_models, max_extraction_time)
    else:        
        persist_results(DATASET, result, stats, path_for_results_file, path_for_framework_models, max_extraction_time, log_to_wandb)
    if log_to_wandb: wandb.finish()

def run():
  params = dict()
  time = 1500
  max_sequence_length = 2
  run_ensemble = False
  use_cache = True
  use_sampling_teacher = True
  log_to_wandb = True

  params[1] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[2] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[3] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[4] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[5] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[6] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[7] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[8] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[9] = {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":1000}
  params[10]= {"max_extraction_time":time, "partitions":100, "max_sequence_len":max_sequence_length, "min_sequence_len":2, "epsilon":0.01, "delta":0.01, "max_states":1000000, "max_query_length":500}

  datasets_to_run = list(range(10))
  path_for_framework_models = "./extraction_results"
  path_for_results_file = get_path_for_result_file_name(path_for_framework_models)
  for ds in datasets_to_run:
    try:
          run_instance(ds, path_for_results_file, path_for_framework_models, params[ds], ensemble = run_ensemble, use_cache=use_cache, use_sampling_teacher = use_sampling_teacher, log_to_wandb = log_to_wandb)        
    except Exception as e:
       print("EXPLOTO!")
       traceback.print_exc()
       if log_to_wandb: wandb.finish()
        
if __name__ == '__main__':
    run()  

    