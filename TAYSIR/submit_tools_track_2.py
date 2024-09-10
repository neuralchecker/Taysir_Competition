import zipfile
import time
import tempfile
import logging
from pathlib import Path
from random import choices

import mlflow

logging.getLogger("mlflow").setLevel(logging.DEBUG)


class func_to_class(mlflow.pyfunc.PythonModel):
    def __init__(self, func):
        self.func = func

    def predict(self, context, model_input):
        return self.func(model_input)


def random_input(size):
    alphabet = list(range(0, size))
    end_symbol = alphabet.pop()
    start_symbol = alphabet.pop()
    return [start_symbol] + choices(alphabet, k=25) + [end_symbol]


def save_function(func, alphabet_size, prefix):
    print("Testing function...")
    input = random_input(alphabet_size)
    output = func(input)
    try:
        float(output)
    except ValueError:
        raise ValueError(
            f"The output should be an integer or a float, but we found a {type(output).__name__}"
        )
    print("Test  passed.")

    print("Creating submission...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow_path = Path(tmp_dir)
        zip_path = f"{prefix}_{int(time.time())}.zip"

        code_paths = list(Path().rglob("*.py"))

        mlflow.pyfunc.save_model(
            path=mlflow_path, python_model=func_to_class(func), code_path=code_paths
        )

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for f in mlflow_path.rglob("*"):
                if f.is_file():
                    zip_file.write(f, f.relative_to(mlflow_path))
    print(f"Submission created at {zip_path}.")
    print("You can now submit your model on the competition website.")


if __name__ == "__main__":
    print(random_input(18))
