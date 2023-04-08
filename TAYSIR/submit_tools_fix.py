import zipfile
import time
import tempfile
import logging
from pathlib import Path, PosixPath
from random import choices

import mlflow

logging.getLogger("mlflow").setLevel(logging.DEBUG)


class func_to_class(mlflow.pyfunc.PythonModel):
    def __init__(self, func):
        self._predict = func.predict

    def predict(self, context, model_input):
        return self._predict(model_input[1:])


def save_function(func, alphabet_size, prefix):
    #print('Testing function...')
    alphabet = list(range(0, alphabet_size))
    sample_input = choices(alphabet, k=25)
    output = func.predict(sample_input)
    try:
        float(output)
    except ValueError:
        raise ValueError(
            f'The output should be an integer or a float, but we found a {type(output).__name__}')
    #print('Test  passed.')

    #print('Creating submission...')
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow_path = Path(tmp_dir)
        
        zip_path = f"predicted_models/{prefix}.zip"

        code_paths = list(Path().rglob('fast_pdfa_wrapper.py'))
        #code_paths = [PosixPath('fast_pdfa_wrapper.py')]
          
        mlflow.pyfunc.save_model(
            path=mlflow_path,
            python_model=func_to_class(func),
            code_path=code_paths
        )

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for f in mlflow_path.rglob('*'):
                if f.is_file():
                    zip_file.write(f, f.relative_to(mlflow_path))
    print(f'Submission created at {zip_path}.')
    #print('You can now submit your model on the competition website.')
