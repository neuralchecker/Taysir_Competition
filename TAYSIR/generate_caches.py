from last_token_weights_pickle_dataset_generator import LastTokenWeightsPickleDataSetGenerator
import mlflow
from pytorch_language_model import PytorchLanguageModel
from run_pdfa_extraction import load_model, create_model, get_alphabet_from_sequences
import torch
from multiprocessing import Pool

def load_and_generate_cache(dataset):
    dataset_generator = LastTokenWeightsPickleDataSetGenerator()
    pytorch_model = load_model(dataset)
    alphabet = get_alphabet_from_sequences(dataset)
    target_model = create_model(alphabet, pytorch_model, dataset)
    dataset_generator.genearte_dataset(target_model, 10_000_000, "./data_caches/"+target_model.name,10_000)


def run():    
    datasets_to_run = list(range(1, 11))
    run_in_parallel = True
    if not run_in_parallel:
        for ds in datasets_to_run:
            load_and_generate_cache(ds)
    else:
        with Pool(len(datasets_to_run)) as p:
            p.map(load_and_generate_cache, datasets_to_run)

if __name__ == '__main__':
    torch.set_num_threads(4)
    run()  
