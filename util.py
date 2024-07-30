from enum import Enum
import pandas as pd
import numpy as np

class TaskType(Enum):
    BINCLASS = 1
    MULTICLASS = 2
    REGRESSION = 3

class LossType(Enum):
    BINCE   = 1
    MULCE   = 2
    MSE     = 3
    SUPCON  = 4

class FeatureType(Enum):
    CATEGORICAL = 1
    SCALAR = 2

SCALAR_NUMERIC = "<|scalarnumeric|>"
CATEGORICAL_UNK = "<|unknowncategorical|>" 
SCALAR_UNK = "<|unknownscalar|>"

def split_data_with_train_validate(datafile, validate_split=0.1, test_split=0):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f"./data/{datafile}.csv")

    assert validate_split > 0
    assert test_split >= 0

    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_validate = int(len(df) * validate_split)
    n_test = int(len(df) * test_split)

    # Split the DataFrame
    df_validate = df_shuffled.iloc[:n_validate]
    df_test = None if n_test == 0 else df_shuffled.iloc[n_validate: n_validate + n_test]
    df_train = df_shuffled.iloc[n_validate + n_test: ]

    # Save the DataFrames to separate CSV files
    df_validate.to_csv(f"./data/{datafile}_validate.csv", index=False)
    df_train.to_csv(f"./data/{datafile}_train.csv", index=False)
    if df_test is not None:
        df_test.to_csv(f"./data/{datafile}_test.csv", index=False)
        
def drop_column(datafile, drop_col, move_col=None):
    df = pd.read_csv(f"./data/{datafile}.csv")

    df = df.drop(columns=[drop_col])

    if move_col is not None:
        move_column = df.pop(move_col)
        df[move_col] = move_column
    df.to_csv(f"./data/{datafile}_contrast.csv", index=False)

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    _initialized = False
    def __init__(self):
        super().__init__()
        if not self._initialized:
            self._initialized = True

    @classmethod
    def get_instance(cls):
        return cls._instances[cls] if cls in cls._instances else None
