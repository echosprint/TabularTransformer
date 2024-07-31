import pandas as pd
import torch
import random
from typing import Tuple, Callable, Union, Optional, Dict
from .preprocessor import data_stats, generate_feature_vocab, preprocess, CategoricalStats, ScalarStats
from .tokenizer import Tokenizer
from .util import TaskType, Singleton, CATEGORICAL_UNK, download
import numpy as np


def load_data(datafile: str) -> pd.DataFrame:
    data = f"./data/{datafile}"
    df = pd.read_csv(data)
    return df


class SingletonDataset(Singleton):

    datafile: str
    min_cat_count: int
    feature_vocab_size: int
    num_cols: int
    task_type: TaskType
    target_map: Optional[Dict[str, int]]
    pretext_target_col: Optional[str]

    def __init__(self,
                 datafile: str,
                 max_seq_len: int = 1024,
                 feature_vocab_size=2048,
                 min_cat_count: Union[int, float] = 200,
                 validate_split: float = 0.2,
                 pretext_with_label: bool = True,
                 pretext_target_col: Optional[str] = None,
                 seed: Optional[int] = None,
                 ):
        if self._initialized:  # make sure initialized only once, to reduce load data overhead
            return

        super().__init__()

        self.datafile = datafile
        filename = f"{self.datafile}.csv"
        self.seed = seed
        # read the dataset into memory
        print(f"load dataset from file: {filename}")
        self.dataset = load_data(filename)

        self.pretext_target_col = pretext_target_col

        if pretext_target_col is not None:
            assert pretext_target_col in self.dataset.columns
            assert (
                not pretext_with_label) or self.dataset.columns[-1] != pretext_target_col

            if pretext_with_label:
                self.dataset = self.dataset.iloc[:, :-1]

            assert f"pretext_target_{
                pretext_target_col.strip()}" not in self.dataset.columns
            self.dataset[f"pretext_target_{
                pretext_target_col.strip()}"] = self.dataset[pretext_target_col].copy()

        self.dataset_x, self.dataset_y = self.dataset.iloc[:,
                                                           :-1], self.dataset.iloc[:, [-1]]
        self.data_size = len(self.dataset_x)

        assert min_cat_count > 0
        self.min_cat_count = min_cat_count if isinstance(min_cat_count, int) else int(
            self.data_size * min_cat_count)  # recommend 2%-5% of data points

        self.stats_x = data_stats(
            self.dataset_x, min_cat_count=self.min_cat_count)
        self.stats_y = data_stats(self.dataset_y, min_cat_count=1)

        self.num_cols = len(self.stats_x[1])
        assert self.num_cols == max_seq_len

        self.feature_vocab = generate_feature_vocab(self.stats_x[0])
        self.feature_vocab_size = len(self.feature_vocab)
        assert self.feature_vocab_size <= feature_vocab_size, "feature vocab size too large for the transformer embedding"

        self.tokenizer = Tokenizer(self.feature_vocab, self.stats_x[1])

        # if pretext_target_col is not None:
        #     # after all the stats step, keep the `pretext_target_col` but fill with UNK
        #     # this make sure different pretrain step has same feature vocab
        #     self.dataset[pretext_target_col] = CATEGORICAL_UNK

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
        assert n_validate >= 0

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


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    batch_size: int
    split: str
    apply_power_transform: bool
    remove_outlier: bool
    unk_ratio: Optional[float]
    seed: Optional[int]

    def __init__(self,
                 batch_size: int,
                 split: str,
                 datafile: str,
                 min_cat_count: Union[int, float] = 200,
                 apply_power_transform=True,
                 remove_outlier=False,
                 unk_ratio: Optional[float] = None,
                 max_seq_len: int = 1024,
                 feature_vocab_size=2048,
                 validate_split: float = 0.2,
                 seed: Optional[int] = None,
                 pretext_with_label: bool = True,
                 pretext_target_col: Optional[str] = None,
                 pretext_col_unk_ratio: Optional[float] = None,
                 ):
        super().__init__()
        self.apply_power_transform = apply_power_transform
        self.remove_outlier = remove_outlier
        self.unk_ratio = unk_ratio
        self.seed = seed

        self.batch_size = batch_size  # number of rows per time

        self.pretext_col_unk_ratio = pretext_col_unk_ratio

        self.sdata = SingletonDataset(datafile,
                                      max_seq_len,
                                      feature_vocab_size,
                                      min_cat_count,
                                      validate_split,
                                      pretext_with_label,
                                      pretext_target_col,
                                      seed,
                                      )

        self.split = split
        assert self.split in ("train", "val")
        # assert self.split == "train" or self.unk_ratio is None or self.unk_ratio <= 0

        self.dataset_x = self.sdata.train_dataset_x if self.split == "train" else self.sdata.validate_dataset_x
        self.dataset_y = self.sdata.train_dataset_y if self.split == "train" else self.sdata.validate_dataset_y

        self.rng = random.Random(self.seed)
        # print(f"Created a PretokDataset with rng seed {self.seed}")

    def __iter__(self):

        dataset_size = len(self.dataset_x)
        num_batches = dataset_size // self.batch_size
        assert num_batches > 0, "this dataset is too small? investigate."

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
                            self.sdata.stats_x[1],
                            self.sdata.stats_x[0],
                            self.apply_power_transform,
                            self.remove_outlier,
                            self.unk_ratio,
                            self.sdata.pretext_target_col,
                            self.pretext_col_unk_ratio,
                            )

            x_tok = self.sdata.tokenizer.encode(xp)
            y_tok = y if self.sdata.target_map is None else y.map(
                lambda x: self.sdata.target_map[x])

            target_dtype = torch.float32 if self.sdata.task_type == TaskType.REGRESSION else torch.long

            yield x_tok, torch.tensor(y_tok.to_numpy(), dtype=target_dtype).squeeze()


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(batch_size, **dataset_kwargs)

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
    def get_dataset_attributes():
        sdata_instance: SingletonDataset = SingletonDataset.get_instance()
        assert sdata_instance is not None
        dataset_att = {
            "feature_vocab_size": sdata_instance.feature_vocab_size,
            "feature_vocab": sdata_instance.feature_vocab,
            "feature_stats": sdata_instance.stats_x[0],
            "feature_type":  sdata_instance.stats_x[1],
            "num_cols":      sdata_instance.num_cols,
            "task_type":     sdata_instance.task_type,
            "target_map":    sdata_instance.target_map,
            "train_dataset_size":  sdata_instance.train_dataset_size,
            "validate_dataset_size":  sdata_instance.validate_dataset_size,
        }
        return dataset_att

    @staticmethod
    def download_dataset(url: str, save_fname: str):
        download(url, save_fname)
