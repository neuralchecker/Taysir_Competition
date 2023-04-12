
from last_token_weights_pickle_dataset_generator import LastTokenWeightsPickleDataSetGenerator
import mlflow
from pytorch_language_model import PytorchLanguageModel
from run_pdfa_extraction import load_model, create_model, get_alphabet_from_sequences

def run():
    dataset_generator = LastTokenWeightsPickleDataSetGenerator()
    datasets_to_run = list(range(1, 11))
    for ds in datasets_to_run:
        pytorch_model = load_model(ds)
        alphabet = get_alphabet_from_sequences(ds)
        target_model = create_model(alphabet, pytorch_model, ds)
        dataset_generator.genearte_dataset(target_model, 1000, "./data_caches/"+target_model.name,1000)

if __name__ == '__main__':
    run()  
