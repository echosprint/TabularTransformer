from abc import ABC, ABCMeta, abstractmethod
import random
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
import sys
from ast import literal_eval
import requests
from tqdm import tqdm
import os
from dataclasses import asdict, fields
from typing import Literal, get_type_hints
from pathlib import Path


class ReaderMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        # Wrap the read_data_file method if it exists
        original_read_data_file = dct.get('read_data_file')
        if original_read_data_file:
            def new_read_data_file(self, *args, **kwargs):
                self.pre_read_data()
                if not args and not kwargs:
                    args = (getattr(self, 'file_path'),)
                result = original_read_data_file(self, *args, **kwargs)
                self.post_read_data(result)
                return result
            dct['read_data_file'] = new_read_data_file
        return super().__new__(cls, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        # Create a new instance of the class
        instance = super().__call__()
        assert not (len(args) == 0 and len(kwargs) == 0), f"""{
            cls} need at least one argument for `file_path`."""
        # Iterate over positional arguments and assign them to attributes
        for i, arg in enumerate(args):
            setattr(instance, f'arg{i}', arg)

        # Iterate over keyword arguments and assign them to attributes
        for key, value in kwargs.items():
            setattr(instance, key, value)

        if not hasattr(instance, 'file_path'):
            if len(args) > 0:
                setattr(instance, 'file_path', args[0])
            else:
                raise ValueError(f"""bad arguments for {
                                 cls}, accept one positional argument or `file_path` keyword argument""")
        instance.file_path = Path(instance.file_path)

        return instance


class DataReader(metaclass=ReaderMeta):
    @abstractmethod
    def read_data_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def ensure_categorical_cols(self):
        pass

    @property
    @abstractmethod
    def ensure_numerical_cols(self):
        pass

    def pre_read_data(self):
        assert isinstance(self.ensure_numerical_cols, list) and (len(self.ensure_numerical_cols) == 0 or all(
            isinstance(e, str) and len(e.strip()) > 0 for e in self.ensure_numerical_cols)), "ensure_numerical_cols must be list of column names"

        assert isinstance(self.ensure_categorical_cols, list) and (len(self.ensure_categorical_cols) == 0 or all(
            isinstance(e, str) and len(e.strip()) > 0 for e in self.ensure_categorical_cols)), "ensure_categorical_cols must be list of column names"

        numerical_set = set(self.ensure_numerical_cols)
        categorical_set = set(self.ensure_categorical_cols)
        common_set = numerical_set.intersection(categorical_set)
        assert len(common_set) == 0, f"""{list(
            common_set)} both in the ensure_numerical_cols and ensure_categorical_cols"""

    def post_read_data(self, df: pd.DataFrame):

        assert isinstance(
            df, pd.DataFrame), "method `read_data_file` must return pd.DataFrame"

        for col in self.ensure_numerical_cols:
            assert col in df.columns, f"""ensure_numerical_cols: `{
                col}` not in data columns"""
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except ValueError as e:
                raise ValueError(
                    f"""Failed to apply `pd.to_numeric` on column [{col}]: {e}""")

        for col in self.ensure_categorical_cols:
            assert col in df.columns, f"""ensure_categorical_cols: `{
                col}` not in data columns"""
            try:
                df[col] = df[col].astype(str)
            except ValueError as e:
                raise ValueError(
                    f"Failed to cast column [{col}] to string: {e}")

    def split_data(self, split: Dict[str, float | int],
                   seed: Optional[int] = 1377,
                   override: bool = True,
                   compression: bool = False) -> Dict[str, Path]:
        assert isinstance(split, dict), "`split` must be Dict[str, float|int]"
        file_path: Path = self.file_path
        base_stem = file_path.stem.split('.')[0]
        suffix = '.csv' if not compression else '.csv.gz'
        if all(file_path.with_name(f"{base_stem}_{sp}{suffix}").exists()
               for sp in split.keys()) \
                and not override:
            print("splits already exists, skip split.")
            return {f'{sp}': file_path.with_name(f"{base_stem}_{sp}{suffix}")
                    for sp in split.keys()}

        print('read data set...')
        data = self.read_data_file()
        data_size = len(data)
        ixs = list(range(data_size))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(ixs)
        start = 0
        fpath = {}
        for sp, ratio in sorted(split.items(), key=lambda kv: -kv[1]):
            assert isinstance(ratio, (float, int))
            assert not isinstance(ratio, int) or ratio == -1 or ratio > 0, \
                "integer split ratio can be -1 or positive intergers, -1 means all the rest of data"
            assert not isinstance(ratio, float) or 1 > ratio > 0, \
                "float split ratio must be interval (0, 1)"
            if isinstance(ratio, int):
                part_len = data_size - start if ratio == -1 else ratio
                assert part_len > 0, f'`no data left for `{sp}` split'
            else:
                part_len = int(data_size * ratio)
                assert part_len > 0, f'`{sp}` split {ratio} two small'
            end = start + part_len
            assert end <= data_size, "bad split: all split sum exceed the data size"
            data_part = data.iloc[ixs[start: end]]
            print(f'split: {sp}, n_samples: {part_len}')

            part_path = file_path.with_name(f"{base_stem}_{sp}{suffix}")

            if part_path.exists() and override:
                os.remove(part_path)
                print(f"{part_path} *exists*, delete old split `{sp}`")

            if not part_path.exists():
                print(f"save split `{sp}` at path: {part_path}")
                data_part.to_csv(part_path, index=False)
            else:
                print(f"{part_path} *exists*, skip split `{sp}`")

            fpath[f'{sp}'] = part_path
            start = end
        return fpath


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


class TypeCheckMeta(type):
    def __call__(cls, *args, **kwargs):
        # Check if any positional arguments are passed
        if args:
            raise TypeError(f"{cls} only accepts keyword arguments.")

        # Get the field definitions with type hints
        field_defs = {f.name: f.type for f in fields(cls)}

        all_args = {**kwargs}

        # Perform type checking
        for name, val in all_args.items():
            expect_type = field_defs.get(name)
            assert expect_type is not None, f"bad {cls} arguments `{name}`"
            if expect_type is float and isinstance(val, int):
                val = float(val)
            # Special case for Literal
            if hasattr(expect_type, "__origin__") and expect_type.__origin__ is Literal:
                assert val in expect_type.__args__ and isinstance(
                    val, type(expect_type.__args__[0])
                ), f"{val} not in {expect_type.__args__}."
            elif hasattr(expect_type, "__origin__") and expect_type.__origin__ is dict:
                assert all(isinstance(key, expect_type.__args__[0]) and isinstance(
                    value, expect_type.__args__[1]) for key, value in val.items()
                ), f"{val} must be type {expect_type}"
            else:
                assert isinstance(
                    val, expect_type
                ), f"{cls} init parameter type mismatch, key: ({name}) expect type: {expect_type}, pass value: {val}"

        # Call the original __init__ method
        return super().__call__(*args, **kwargs)


class DataclassTool(metaclass=TypeCheckMeta):
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
        return f"{type(self).__name__}: {asdict(self)}"

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
