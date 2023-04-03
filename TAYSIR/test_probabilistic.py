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
    DATASET = 2
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
    
    LastTokenWeightsPickleDataSetGenerator().genearte_dataset(target_model, 100, "./test",10)
    
if __name__ == '__main__':
    run()