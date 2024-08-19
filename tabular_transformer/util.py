from enum import Enum
import pandas as pd
import warnings
import os
import requests


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

    fdict1 = {k: v for k, v in dict1.items() if k not in ignore_key}
    fdict2 = {k: v for k, v in dict2.items() if k not in ignore_key}

    union_keys = set(fdict1.keys()) | set(fdict2.keys())

    diff_dict = {key: (fdict1.get(key, None),
                       fdict2.get(key, None))
                 for key in union_keys
                 if fdict1.get(key, None) !=
                 fdict2.get(key, None)}

    # Compare the filtered dictionaries
    return fdict1 == fdict2, diff_dict


def prepare_income_dataset():
    website = "https://huggingface.co/datasets/scikit-learn/adult-census-income"
    data_url = "hf://datasets/scikit-learn/adult-census-income/adult.csv"
    fname = "income.csv"
    income_path = prepare_dataset(data_url, fname, website)
    return income_path


def prepare_fish_dataset():
    website = "https://huggingface.co/datasets/scikit-learn/Fish"
    data_url = "hf://datasets/scikit-learn/Fish/Fish.csv"
    fname = "fish.csv"
    fish_path = prepare_dataset(data_url, fname, website)
    return fish_path


def prepare_dataset(data_url, fname, website):
    warnings.filterwarnings('ignore', category=UserWarning)

    data_cache_dir = os.path.join(os.getcwd(), 'data', fname.split('.')[0])
    os.makedirs(data_cache_dir, exist_ok=True)
    full_path = os.path.join(data_cache_dir, fname)

    print(f"more details see website: {website}")
    if not os.path.exists(full_path):
        print(f"Downloading {data_url} to {fname} ...")
        df = pd.read_csv(data_url)
        df.to_csv(full_path, index=False)
        print(f"save data at path: {full_path}")
    else:
        df = pd.read_csv(full_path)
        print(f"{full_path} already exists, skipping download.")
    warnings.filterwarnings('default', category=UserWarning)
    return full_path


def download_files_from_github(repo, folder_path, local_dir):
    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Construct the GitHub API URL
    api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
    response = requests.get(api_url)
    response.raise_for_status()

    # Get the content of the folder
    files = response.json()

    for file in files:
        if file['type'] == 'file':
            download_url = file['download_url']
            file_name = file['name']
            local_file_path = os.path.join(local_dir, file_name)

            # Download the file
            print(f"Downloading {file_name}...")
            file_response = requests.get(download_url)
            file_response.raise_for_status()

            # Save the file locally
            with open(local_file_path, 'wb') as local_file:
                local_file.write(file_response.content)

            print(f"Saved {file_name} to {local_file_path}")

        elif file['type'] == 'dir':
            # Recursively download files in subdirectories
            download_files_from_github(
                repo, f"{folder_path}/{file['name']}", f"{local_dir}/{file['name']}")


def download_notebooks():

    repo = "echosprint/TabularTransformer"
    folder_path = "notebooks/"  # Replace with the folder path in the repo
    # Replace with the desired local directory to save files
    local_dir = "./notebooks/"

    download_files_from_github(repo, folder_path, local_dir)
