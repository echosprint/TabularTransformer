from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import asdict
from typing import Literal, get_type_hints
import sys
from ast import literal_eval


class TaskType(Enum):
    BINCLASS = 1
    MULTICLASS = 2
    REGRESSION = 3


class LossType(Enum):
    BINCE = 1
    MULCE = 2
    MSE = 3
    SUPCON = 4


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
    df_train = df_shuffled.iloc[n_validate + n_test:]

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


class DataclassTool:
    def __init__(self):
        raise NotImplementedError("DataclassTool should not be instantiated.")

    def update(self, hypara: str, val):
        if hypara in asdict(self):
            # ensure the types match
            expect_type = get_type_hints(self)[hypara]

            if expect_type is float and isinstance(val, int):
                val = float(val)

            if expect_type is bool and isinstance(val, str):
                if val.lower() == "false":
                    val = False
                elif val.lower() == "true":
                    val = True

            # Special case for Literal
            if hasattr(expect_type, "__origin__") and expect_type.__origin__ is Literal:
                assert val in expect_type.__args__ and isinstance(
                    val, type(expect_type.__args__[0])
                ), f"{val} not in {expect_type.__args__}."
            else:
                assert isinstance(
                    val, expect_type
                ), f"hyperparameter type mismatch, key: ({hypara}) expect type: {expect_type}, pass value: {val}"

            print(f"Overriding hyperparameter: {hypara} = {val}")
            setattr(self, hypara, val)
        else:
            raise ValueError(f"Unknown config hyperparameter key: {hypara}")

    def __str__(self):
        return f"HyperParameters: {asdict(self)}"

    def asdict(self):
        return asdict(self)

    def config_from_cli(self):
        for arg in sys.argv[1:]:
            # assume it's a --key=value argument
            assert arg.startswith(
                '--'), f"specify hyperparameters must in --key=value format"
            key, val = arg.split('=')
            key = key[2:]  # skip --

            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val

            self.update(key, attempt)
