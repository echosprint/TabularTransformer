from enum import Enum
import pandas as pd


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
    NUMERICAL = 2


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


def equals_except(dict1, dict2, ignore_key):
    assert isinstance(dict1, dict) and isinstance(
        dict2, dict), "Both inputs must be dictionaries."

    assert isinstance(ignore_key, (str, tuple, list)
                      ), "Ignore key must be a string, tuple, or list."

    if isinstance(ignore_key, str):
        # Convert to a list for consistent processing
        ignore_key = [ignore_key]

    # Create new dictionaries excluding the keys to ignore
    filtered_dict1 = {k: v for k, v in dict1.items() if k not in ignore_key}
    filtered_dict2 = {k: v for k, v in dict2.items() if k not in ignore_key}

    # Compare the filtered dictionaries
    return filtered_dict1 == filtered_dict2
