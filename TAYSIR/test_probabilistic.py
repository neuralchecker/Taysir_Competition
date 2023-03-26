import torch
import mlflow
print('PyTorch version :', torch.__version__)
print('MLflow version :', mlflow.__version__)
import sys
print("Your python version:", sys.version)

def predict(sequence, model):
    """ Define the function that takes a sequence as a list of integers and return the decision of your extracted model. 
    The function does not need your model as argument. See baseline notebooks for examples."""
    
    #For instance, if you want to submit the original model:
    try: #RNN
        value = model.predict(model.one_hot_encode(sequence)) 
        return value
    except: #Transformer
        def make_future_masks(words:torch.Tensor):
            masks = (words != 0)
            b,l = masks.size()
            #x = einops.einsum(masks, masks, "b i, b j -> b i j")
            x = torch.einsum("bi,bj->bij",masks,masks)
            x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril()
            x += torch.eye(l,dtype=torch.bool, device=x.device)
            return x.type(torch.int8)
        import numpy
        def predict_next_symbols(model, word):
            """
            Args:
                whole word (list): a complete sequence as a list of integers
            Returns:
                the predicted probabilities of the next ids for all prefixes (2-D ndarray)
            """
            word = [ [ a+1 for a in word ] ]
            word = torch.IntTensor(word)
            model.eval()
            with torch.no_grad():
                attention_mask = make_future_masks(word)
                out = model.forward(word, attention_mask=attention_mask)
                out = torch.nn.functional.softmax(out.logits[0], dim=1)
                return out.detach().numpy()[:, 1:] #  the probabilities for padding id (0) are removed
        def predict_transformer(model, word):
            probs = predict_next_symbols(model, word[:-1])
            probas_for_word = [probs[i,a] for i,a in enumerate(word[1:])]
            value = numpy.array(probas_for_word).prod()
            return float(value)
        return predict_transformer(model, sequence)
    
def run():
    TRACK = 2 #always for this track
    DATASET = 1

    model_name = f"models/2.{DATASET}.taysir.model"

    model = mlflow.pytorch.load_model(model_name)
    model.eval()

    try:#RNN
        nb_letters = model.input_size -1
        cell_type = model.cell_type

        print("The alphabet contains", nb_letters, "symbols.")
        print("The type of the recurrent cells is", cell_type.__name__)
    except:
        nb_letters = model.distilbert.config.vocab_size
        print("The alphabet contains", nb_letters, "symbols.")
        print("The model is a transformer (DistilBertForSequenceClassification)")

    file = f"datasets/2.{DATASET}.taysir.valid.words"

    sequences = []
    with open(file) as f:
        f.readline() #Skip first line (number of sequences, alphabet size)
        for line in f:
            line = line.strip()
            seq = line.split(' ')
            seq = [int(i) for i in seq[1:]] #Remove first value (length of sequence) and cast to int
            sequences.append(seq)

    print('Number of sequences:', len(sequences))
    print('10 first sequences:')
    for i in range(10):
        print(sequences[i])

    print(predict(sequences[1], model))


if __name__ == '__main__':
    run()