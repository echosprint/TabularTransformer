from tabular_transformer.tabular_dataset import TabularDataset
from typing import Dict, Optional
import torch


class TokenDataset(torch.utils.data.IterableDataset):
    """Loads tabular data PyTorch tensors."""

    batch_size: int
    split: str
    tabular_dataset: TabularDataset
    num_rows: int
    unk_ratio: Optional[Dict[str, float]]
    unk_ratio_default: Optional[float]
    seed: Optional[int]
    device: str
    enable_unk: bool

    def __init__(self,
                 batch_size: int,
                 split: str,
                 tabular_dataset: TabularDataset,
                 unk_ratio: Optional[Dict[str, float]] = None,
                 unk_ratio_default: Optional[float] = None,
                 seed: Optional[int] = 42,
                 ):
        super().__init__()
        self.unk_ratio = unk_ratio
        self.unk_ratio_default = unk_ratio_default

        self.device = tabular_dataset.device

        self.seed = seed
        self.rng = torch.Generator(device=self.device)
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self.batch_size = batch_size  # number of rows per time

        self.tabular_dataset = tabular_dataset
        self.num_rows = self.tabular_dataset.num_rows

        self.split = split
        assert self.split in ("train", "val")

        if self.split == "train":
            self.split_indices = self.tabular_dataset.train_split_indices
        else:
            self.split_indices = self.tabular_dataset.validate_split_indices
        assert self.split_indices.size(0) > 0, \
            f"the `{self.split}` split of dataset is empty."

        self.dataset_x_tok = self.tabular_dataset.dataset_x_tok
        self.dataset_x_val = self.tabular_dataset.dataset_x_val
        self.dataset_y = self.tabular_dataset.dataset_y

        self.dataset_split_size = self.split_indices.size(0)
        self.num_batches = self.dataset_split_size // self.batch_size

        assert self.batch_size <= self.dataset_split_size, \
            f"""the batch size is too large for the dataset, batch_size: {
                self.batch_size}, dataset `{self.split}` split size: {self.dataset_split_size}"""

        self.init_unk_mask()

    def init_unk_mask(self):

        if self.unk_ratio is not None:
            assert self.unk_ratio_default is not None
            assert isinstance(self.unk_ratio, dict)
            assert all(col in self.tabular_dataset.column_names
                       and 0 <= val <= 1
                       for col, val in self.unk_ratio.items()), \
                "col in `unk_ratio` not exist."

        if self.unk_ratio_default is None:
            self.enable_unk = False
            self.unk_threshold = torch.tensor(0., device=self.device)
            return

        assert 0 <= self.unk_ratio_default <= 1
        self.enable_unk = True

        unk_ratio = {} if self.unk_ratio is None else self.unk_ratio
        unk_ratio_arr = [unk_ratio.get(col, self.unk_ratio_default)
                         for col, _ in self.tabular_dataset.feature_stats.x_col_type]
        self.unk_threshold = torch.tensor(unk_ratio_arr, device=self.device)

    def mask_unk(self, x_tok, x_val):

        if not self.enable_unk:
            return x_tok, x_val

        unk_mask = torch.rand(
            x_tok.shape, device=self.device) >= self.unk_threshold

        x_tok_unk = torch.where(unk_mask,
                                x_tok,
                                self.tabular_dataset.unk_tok.unsqueeze(0).repeat(
                                    x_tok.shape[0], 1)
                                )
        x_val_unk = torch.where(unk_mask,
                                x_val,
                                torch.ones(x_val.shape,
                                           device=self.device)
                                )
        return x_tok_unk, x_val_unk

    def __iter__(self):

        while True:

            ixs = torch.randperm(
                self.dataset_split_size,
                device=self.device,
                generator=self.rng
            )

            for n in range(self.num_batches):
                start = n * self.batch_size
                end = start + self.batch_size

                indices = self.split_indices[ixs[start: end]]
                x_tok = self.dataset_x_tok[indices]
                x_val = self.dataset_x_val[indices]
                y = self.dataset_y[indices]

                x_tok, x_val = self.mask_unk(x_tok, x_val)

                x_tok = x_tok.long()
                y = y if torch.is_floating_point(y) else y.long()

                yield (x_tok, x_val), y


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = TokenDataset(batch_size, **dataset_kwargs)

        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, pin_memory=True, num_workers=num_workers)

        def generator():
            for (x_tok, x_val), y in dl:
                x_tok = x_tok.to(device, non_blocking=True)
                x_val = x_val.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                yield (x_tok, x_val), y
        return generator()
