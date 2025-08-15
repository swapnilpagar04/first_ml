import os
import sys
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        print(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    """
    Load an object from a file using dill.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
