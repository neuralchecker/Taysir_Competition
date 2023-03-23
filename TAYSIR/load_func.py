import zipfile
import mlflow
import shutil

def load_function(path_to_zip, validation_data):

    # unzip in tmp folder
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall('tmp')

    # load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri='tmp/')
    unwrapped_model = loaded_model.unwrap_python_model()

    # execute few sequences to check if loading worked
    print('Model loaded, testing it on 10 sequences')
    for seq in validation_data[:10]:
        # Context is None?
        o = unwrapped_model.predict(None, seq)
        print(o)

    # remove tmp subfolder
    shutil.rmtree('tmp/')