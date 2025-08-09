import os
import sys
import numpy as np
import pandas as pd 
import dill


from src.exeption import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using dill.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        print(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e