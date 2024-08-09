import pandas as pd
import torch
import random
from typing import Tuple, Callable, Union, Optional, Dict
from .preprocessor import data_stats, generate_feature_vocab, preprocess, CategoricalStats, NumericalStats
from .tokenizer import Tokenizer
from .util import FeatureType, TaskType, CATEGORICAL_UNK
from .data_common import DataReader, download
import numpy as np


def load_data(datafile: str) -> pd.DataFrame:
    data = f"./data/{datafile}"
    df = pd.read_csv(data)
    return df


class RawDataset():

    datareader: DataReader
    min_cat_count: int
    num_cols: int
    task_type: TaskType
    target_map: Optional[Dict[str, int]]
    n_validate: int

    def __init__(self,
                 datareader: DataReader,
                 min_cat_count: Union[int, float] = 200,
                 validate_split: float = 0.2,
                 seed: Optional[int] = None,
                 ):

        self.datareader = datareader
        self.seed = seed
        # read the dataset into memory
        print(f"load dataset from file: {self.datareader.file_path}")
        self.dataset = self.datareader.read_data_file()

        self.dataset_x, self.dataset_y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, [-1]]  # noqa: E501
        self.data_size = len(self.dataset_x)

        assert min_cat_count > 0
        self.min_cat_count = min_cat_count if isinstance(min_cat_count, int) else int(
            self.data_size * min_cat_count)  # recommend 2%-5% of data points

        self.stats_x = data_stats(
            self.dataset_x, min_cat_count=self.min_cat_count)

        self.feature_stats, self.feature_type = self.stats_x

        self.stats_y = data_stats(self.dataset_y, min_cat_count=1)

        self.num_cols = len(self.stats_x[1])

        self.feature_vocab = generate_feature_vocab(self.stats_x[0])
        self.feature_vocab_size = len(self.feature_vocab)

        self.train_dataset, self.validate_dataset = self._split_dataset(
            validate_split=validate_split)

        self.train_dataset_size, self.validate_dataset_size = len(
            self.train_dataset), len(self.validate_dataset)

        self.train_dataset_x, self.train_dataset_y = self.train_dataset.iloc[
            :, :-1], self.train_dataset.iloc[:, [-1]]
        self.validate_dataset_x, self.validate_dataset_y = self.validate_dataset.iloc[
            :, :-1], self.validate_dataset.iloc[:, [-1]]

        self._get_tasktype()

    def _split_dataset(self, validate_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Shuffle the DataFrame
        dataset_shuffled = self.dataset.sample(
            frac=1.0, replace=False, random_state=self.seed)

        n_validate = int(len(self.dataset) * validate_split)
        assert len(self.dataset) > n_validate >= 0
        self.n_validate = n_validate
        # Split the DataFrame
        dataset_validate = dataset_shuffled.iloc[:n_validate]
        dataset_train = dataset_shuffled.iloc[n_validate:]
        return dataset_train, dataset_validate

    def _get_tasktype(self):
        target_stats = list(self.stats_y[0].values())[0]
        if isinstance(target_stats, CategoricalStats):
            if len(target_stats.valid_cats) == 2:
                self.task_type = TaskType.BINCLASS
            elif len(target_stats.valid_cats) > 2:
                self.task_type = TaskType.MULTICLASS

            self.target_map = {key: index for index,
                               key in enumerate(target_stats.valid_cats.keys())}
        else:
            self.task_type = TaskType.REGRESSION
            self.target_map = None
        return self.task_type

    def dataset_attr(self):
        dataset_att = {
            "feature_vocab_size": self.feature_vocab_size,
            "feature_vocab": self.feature_vocab,
            "feature_stats": self.stats_x[0],
            "feature_type":  self.stats_x[1],
            "num_cols":      self.num_cols,
            "task_type":     self.task_type,
            "target_map":    self.target_map,
            "train_dataset_size":  self.train_dataset_size,
            "validate_dataset_size":  self.validate_dataset_size,
        }
        return dataset_att


class TabularDataset(torch.utils.data.IterableDataset):
    """Loads tabular samples from disk and yields them as PyTorch tensors."""

    batch_size: int
    split: str
    raw_dataset: RawDataset
    apply_power_transform: bool
    remove_outlier: bool
    unk_ratio: Optional[Dict[str, float]]
    unk_ratio_default: Optional[float]
    feature_type: Dict[str, FeatureType]
    feature_stats: Dict[str, Union[CategoricalStats, NumericalStats]]
    tokenizer: Tokenizer
    target_map: Dict[str, int]
    task_type: TaskType
    seed: Optional[int]

    def __init__(self,
                 batch_size: int,
                 split: str,
                 raw_dataset: RawDataset,
                 feature_type: Dict[str, FeatureType],
                 feature_stats: Dict[str, Union[CategoricalStats, NumericalStats]],
                 tokenizer: Tokenizer,
                 target_map: Dict[str, int],
                 task_type: TaskType,
                 apply_power_transform=True,
                 remove_outlier=False,
                 unk_ratio: Optional[Dict[str, float]] = None,
                 unk_ratio_default: Optional[float] = None,
                 seed: Optional[int] = None,
                 ):
        super().__init__()
        self.apply_power_transform = apply_power_transform
        self.remove_outlier = remove_outlier
        self.unk_ratio = unk_ratio
        self.unk_ratio_default = unk_ratio_default
        self.seed = seed

        self.batch_size = batch_size  # number of rows per time

        self.raw_dataset = raw_dataset
        self.feature_type = feature_type
        self.feature_stats = feature_stats
        self.tokenizer = tokenizer
        self.target_map = target_map
        self.task_type = task_type

        self.split = split
        assert self.split in ("train", "val")

        self.dataset_x = self.raw_dataset.train_dataset_x if self.split == "train" else self.raw_dataset.validate_dataset_x
        self.dataset_y = self.raw_dataset.train_dataset_y if self.split == "train" else self.raw_dataset.validate_dataset_y

        self.rng = random.Random(self.seed)

    def __iter__(self):

        dataset_size = len(self.dataset_x)
        num_batches = dataset_size // self.batch_size

        assert dataset_size > 0, "dataset is empty"
        assert self.batch_size <= dataset_size, "the batch size is too large for the dataset"

        ixs = list(range(dataset_size))
        self.rng.shuffle(ixs)

        for n in range(num_batches):
            start = n * self.batch_size
            end = start + self.batch_size

            x = self.dataset_x.iloc[ixs[start: end]]
            y = self.dataset_y.iloc[ixs[start: end]]

            # preprocess the data
            xp = preprocess(self.rng,
                            x,
                            self.feature_type,
                            self.feature_stats,
                            self.apply_power_transform,
                            self.remove_outlier,
                            self.unk_ratio,
                            self.unk_ratio_default,
                            )

            x_tok = self.tokenizer.encode(xp)
            y_tok = y if self.target_map is None else y.map(
                lambda x: self.target_map[x])

            target_dtype = torch.float32 if self.task_type == TaskType.REGRESSION else torch.long

            yield x_tok, torch.tensor(y_tok.to_numpy(), dtype=target_dtype).squeeze()


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = TabularDataset(batch_size, **dataset_kwargs)

        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, pin_memory=True, num_workers=num_workers)

        def generator():
            for x, y in dl:
                feature_tokens = x[0].to(device, non_blocking=True)
                feature_weight = x[1].to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                yield (feature_tokens, feature_weight), y
        return generator()

    @staticmethod
    def download_dataset(url: str, save_fname: str):
        download(url, save_fname)
