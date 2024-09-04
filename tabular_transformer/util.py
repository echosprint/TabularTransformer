from enum import Enum
import zipfile
import pandas as pd
import warnings
import os
import requests
import requests
from tqdm import tqdm


class TaskType(Enum):
    BINCLASS = 1
    MULTICLASS = 2
    REGRESSION = 3


class LossType(Enum):
    BINCE = 1
    MULCE = 2
    MSE = 3
    SUPCON = 4


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


def prepare_iris_dataset():
    website = "https://huggingface.co/datasets/scikit-learn/iris"
    data_url = "hf://datasets/scikit-learn/iris/Iris.csv"
    fname = "iris.csv"
    iris_path = prepare_dataset(data_url, fname, website)
    return iris_path


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


def prepare_higgs_dataset():
    fname = 'higgs.csv.gz'
    url = "https://archive.ics.uci.edu/static/public/280/higgs.zip"

    data_cache_dir = os.path.join(os.getcwd(), 'data', fname.split('.')[0])
    os.makedirs(data_cache_dir, exist_ok=True)
    extracted_file_path = os.path.join(data_cache_dir, fname)

    if os.path.exists(extracted_file_path):
        print("higgs dataset exists, skip download and extraction.")
        return extracted_file_path

    zip_file_path = os.path.join(data_cache_dir, 'higgs.zip')

    if not os.path.exists(zip_file_path):
        print("Downloading the Higgs dataset...")
        response = requests.get(url)
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)
        print("Download completed.")
    else:
        print("higgs zip file exists, skip downloading.")

    print("Extracting the ZIP file...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_cache_dir)
    print("ZIP extraction completed.")

    # Find and extract the .gz file
    csv_gz = False
    for root, dirs, files in os.walk(data_cache_dir):
        for file in files:
            if file.endswith('.csv.gz'):
                gz_file_path = os.path.join(root, file)
                os.rename(gz_file_path, extracted_file_path)
                csv_gz = True
                break
                # print(f"Extracting {gz_file_path}...")
                # with gzip.open(gz_file_path, 'rb') as f_in:
                #     with open(extracted_file_path, 'wb') as f_out:
                #         shutil.copyfileobj(f_in, f_out)
                # print(f"Extraction of {gz_file_path} completed.")

                # # Optionally, remove the .gz file after extraction
                # os.remove(gz_file_path)
                # print(f"Removed {gz_file_path}.")

        if csv_gz:
            break
    # Clean up the zip file
    os.remove(zip_file_path)
    print("Cleanup completed. Dataset is ready.")
    print(f"Higgs dataset saved: {extracted_file_path}")
    return extracted_file_path


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(url: str, fname: str):
    """Downloads the dataset to DATA_CACHE_DIR"""

    DATA_CACHE_DIR = os.path.join(
        os.path.dirname(__file__), 'data', fname.split('.')[0])
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset, unless it's already downloaded
    data_url = url
    data_filename = os.path.join(DATA_CACHE_DIR, fname)
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # # unpack the tar.gz file into all the data shards (json files)
    # data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir, exist_ok=True)
    #     print(f"Unpacking {data_filename}...")
    #     os.system(f"tar -xzf {data_filename} -C {data_dir}")
    # else:
    #     print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    # shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    # print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")
