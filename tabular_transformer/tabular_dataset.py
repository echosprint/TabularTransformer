from typing import Dict, List, Optional, Union
import numpy as np
import pyarrow.compute as pc
import pyarrow as pa
import torch

from tabular_transformer.datareader import DataReader
from tabular_transformer.featurestats import FeatureStats
from tabular_transformer.util import TaskType


class TabularDataset():
    datareader: DataReader

    validate_split: Optional[float]

    seed: Optional[int]

    id: Optional[np.ndarray]

    min_cat_count: int
    task_type: Optional[TaskType]
    n_class: Optional[int]

    apply_power_transform: bool
    device: str

    original_feature_stats: Optional[FeatureStats]
    feature_stats: FeatureStats
    merged_feature_stats: FeatureStats

    # num of features x, `label` not included
    num_cols: int
    # num of samples
    num_rows: int
    column_names: List[str]

    # label column name
    label: Optional[str]
    ensure_categorical_cols: List[str]
    ensure_numerical_cols: List[str]

    # dataset for loader
    dataset_x_tok: torch.Tensor
    dataset_x_val: torch.Tensor
    dataset_y: Optional[torch.Tensor]

    n_train: int
    n_validate: int
    train_split_indices: torch.Tensor
    validate_split_indices: torch.Tensor

    def __init__(self,
                 datareader: DataReader,
                 device: str = 'cpu',
                 original_feature_stats: FeatureStats = None,
                 min_cat_count: Union[int, float] = 200,
                 apply_power_transform: bool = True,
                 validate_split: Optional[float] = 0.2,
                 seed: Optional[int] = 42,
                 ):
        self.device = device
        self.apply_power_transform = apply_power_transform

        self.seed = seed
        self.rng = torch.Generator(device=self.device)
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self.validate_split = validate_split

        self.datareader = datareader
        table = self.datareader.read()
        self.label = self.datareader.label
        self.ensure_categorical_cols = self.datareader.ensure_categorical_cols
        self.ensure_numerical_cols = self.datareader.ensure_numerical_cols
        self.num_rows = table.num_rows
        self.column_names = table.column_names

        self.original_feature_stats = original_feature_stats
        self.feature_stats = FeatureStats()

        assert min_cat_count > 0
        self.min_cat_count = min_cat_count if isinstance(min_cat_count, int) else int(
            self.num_rows * min_cat_count)  # recommend 2%-5% of data points

        table_x, table_y = self.extract_table(table)

        table_x = self.mask_id_column(table_x)
        self.stat_col_type_x(table_x)
        self.stat_cat_cls_x(table_x)
        tok_tensor, val_tensor = self.process_table_x(table_x)
        self.stat_num_x(tok_tensor, val_tensor)
        self.transform_x(tok_tensor, val_tensor)

        self.stats_transform_y(table_y)

        self.merged_feature_stats = self.feature_stats.merge_original(
            self.original_feature_stats)

        self.split_dataset()

        del table, table_x, table_y
        del tok_tensor, val_tensor

        if 'cuda' in self.device:
            torch.cuda.empty_cache()

    def extract_table(self, table):
        if self.label is None:
            return table, None
        assert self.label in table.column_names, \
            f"`label` '{self.label}' not exists"
        column_x = [
            col for col in table.column_names if col != self.label]
        column_y = [self.label]
        return table.select(column_x), table.select(column_y)

    def mask_id_column(self, table: pa.Table):
        id = self.datareader.id
        if id is None:
            self.id = None
            return table

        self.id = table.column(id).to_numpy()

        mask_id_col = pa.array(np.full(self.num_rows, 'mask_id'))

        mask_id_table = table.set_column(
            table.column_names.index(id), id, mask_id_col)
        return mask_id_table

    def stat_cat_cls_x(self, table):

        if self.original_feature_stats is not None:
            return

        cls_dict = {}
        for col in table.column_names:
            if col in self.ensure_categorical_cols:
                # col_type.append((col, 'cat'))

                cls_counts = pc.value_counts(table[col])

                valid_cls_counts = {
                    count_struct['values'].as_py(): count_struct['counts'].as_py()
                    for count_struct in cls_counts
                    if count_struct['values'].is_valid and
                    len(str(count_struct['values'])) > 0 and
                    count_struct['counts'].as_py() >= self.min_cat_count
                }

                valid_cls = list(valid_cls_counts.keys())

                assert len(valid_cls) > 0, \
                    f"no class in col `{col}` satisfies `min_cat_count`"

                cls_dict[col] = valid_cls
        self.feature_stats = self.feature_stats(x_cls_dict=cls_dict)

    def stat_col_type_x(self, table):
        col_type = []
        for col in table.column_names:
            if col in self.ensure_categorical_cols:
                col_type.append((col, 'cat'))
            elif col in self.ensure_numerical_cols:
                col_type.append((col, 'num'))
            else:
                raise ValueError("bad col")

        self.feature_stats = self.feature_stats(x_col_type=col_type)
        assert self.original_feature_stats is None or \
            self.original_feature_stats.x_col_type == self.feature_stats.x_col_type, \
            "column type mismatch."

    def process_table_x(self, table):

        table_col_names = table.column_names
        num_cols = len(table_col_names)
        self.num_cols = num_cols
        cls_num = num_cols

        tok_table = []
        val_table = []

        cls_dict = self.feature_stats.x_cls_dict \
            if self.original_feature_stats is None \
            else self.original_feature_stats.x_cls_dict

        for idx, col in enumerate(table_col_names):
            if col in self.ensure_categorical_cols:

                valid_cls = cls_dict[col]

                int_col = pc.index_in(
                    table[col], value_set=pa.array(valid_cls))

                int_col = pc.add(int_col, cls_num)
                int_col = pc.fill_null(int_col, idx)
                int_col = pc.cast(int_col, pa.int16())

                tok_table.append(int_col.to_numpy().astype(np.int16))
                val_table.append(
                    np.full(len(int_col), 1.0, dtype=np.float32))

                cls_num += len(valid_cls)

            elif col in self.ensure_numerical_cols:
                valid_check = pc.is_valid(table[col])
                tok_col = pc.if_else(valid_check, cls_num, idx)
                tok_col = pc.cast(tok_col, pa.int16())
                fill_col = pc.fill_null(table[col], 1.0)

                tok_table.append(tok_col.to_numpy().astype(np.int16))
                val_table.append(fill_col.to_numpy().astype(np.float32))

                cls_num += 1

            else:
                raise ValueError("Bad column name")

            assert max(cls_num, idx) < 32767

        tok_tensor = torch.tensor(np.array(tok_table), dtype=torch.int16)
        val_tensor = torch.tensor(np.array(val_table), dtype=torch.float32)

        tok_tensor = tok_tensor.to(self.device, non_blocking=True)
        val_tensor = val_tensor.to(self.device, non_blocking=True)

        self.col_mask = torch.tensor(
            [False if ty == 'cat' else True
             for _, ty in self.feature_stats.x_col_type], device=self.device
        )

        self.unk_tok = torch.arange(
            self.num_cols, device=tok_tensor.device, dtype=torch.int16)

        return tok_tensor, val_tensor

    def power_transform(self, tensor: torch.Tensor):
        return torch.where(
            tensor < 0, -torch.log1p(-tensor), torch.log1p(tensor))

    def cal_mean_std(self,
                     tok_tensor: torch.Tensor,
                     val_tensor: torch.Tensor,
                     col_mask: torch.Tensor,
                     log_transform: bool = False):

        null_mask = (tok_tensor[col_mask] >
                     self.unk_tok[col_mask].unsqueeze(1)).float()

        val_tensor = val_tensor[col_mask]

        if log_transform:
            val_tensor = self.power_transform(val_tensor)

        non_null_tensor = val_tensor * null_mask

        mean_values = non_null_tensor.sum(dim=1) / null_mask.sum(dim=1)

        squared_errors = (
            (non_null_tensor - mean_values.unsqueeze(1)) ** 2) * null_mask

        # pandas use `n-1` not `n` as denominator for std calculation
        std_values = torch.sqrt(squared_errors.sum(
            dim=1) / (null_mask.sum(dim=1)))

        return mean_values, std_values

    def stat_num_x(self, tok_tensor, val_tensor):
        if self.original_feature_stats is not None:
            return

        mean_values, std_values = self.cal_mean_std(
            tok_tensor, val_tensor, self.col_mask)

        mean_log_values, std_log_values = self.cal_mean_std(
            tok_tensor, val_tensor, self.col_mask, log_transform=True)
        numerical_stats = {"mean": mean_values.cpu(),
                           "std": std_values.cpu(),
                           "mean_log": mean_log_values.cpu(),
                           "std_log": std_log_values.cpu()}
        self.feature_stats = self.feature_stats(x_num_stats=numerical_stats)

    def stats_transform_y(self, table_y):
        if table_y is None:
            self.dataset_y = None
            self.n_class = None
            self.task_type = None
            return

        col_y = table_y.column(0)

        if self.label in self.ensure_categorical_cols:
            y_cls = col_y.unique().to_pylist()
            assert all(cls is not None and len(cls) > 0 for cls in y_cls), \
                "class in label must not be empty"
            self.feature_stats = self.feature_stats(y_cls=y_cls, y_type='cat')
            valid_cls = self.original_feature_stats.y_cls \
                if self.original_feature_stats is not None and \
                self.original_feature_stats.y_cls is not None \
                else self.feature_stats.y_cls
            assert len(valid_cls) >= 2, \
                "there must be at least two classes in label"
            int_col = pc.index_in(
                col_y, value_set=pa.array(valid_cls))

            y_tok = pc.cast(int_col, pa.int16()).to_numpy().astype(np.int16)
            self.dataset_y = torch.tensor(y_tok, device=self.device)
            self.n_class = 1 if len(valid_cls) <= 2 else len(valid_cls)
            self.task_type = TaskType.BINCLASS if self.n_class == 1 else TaskType.MULTICLASS

        elif self.label in self.ensure_numerical_cols:
            assert pc.sum(pc.is_null(col_y, nan_is_null=True)).as_py() == 0, \
                f"Label Column '{self.label}' contains null values."
            y_numpy = col_y.to_numpy()
            y_tensor = torch.tensor(y_numpy, device=self.device)

            y_mean = y_tensor.mean().item()
            y_std = y_tensor.std().item()

            y_log = self.power_transform(y_tensor)

            y_mean_log = y_log.mean().item()
            y_std_log = y_log.std().item()

            self.feature_stats = self.feature_stats(
                y_num_stats=(y_mean, y_std, y_mean_log, y_std_log),
                y_type='num')

            y_num_stats = self.original_feature_stats.y_num_stats \
                if self.original_feature_stats is not None and \
                self.original_feature_stats.y_num_stats is not None \
                else self.feature_stats.y_num_stats

            mean, std, mean_log, std_log = y_num_stats

            if self.apply_power_transform:
                self.dataset_y = (y_log - mean_log) / (std_log + 1e-8)
            else:
                self.dataset_y = (y_tensor - mean) / (std + 1e-8)
            self.n_class = 1
            self.task_type = TaskType.REGRESSION

        else:
            raise ValueError("bad label")

    def transform_x(self, tok_tensor, val_tensor):
        feature_stats = self.feature_stats if self.original_feature_stats is None else self.original_feature_stats
        mean_values = feature_stats.x_num_stats['mean'].to(device=self.device)
        std_values = feature_stats.x_num_stats['std'].to(device=self.device)
        mean_log_values = feature_stats.x_num_stats['mean_log'].to(
            device=self.device)
        std_log_values = feature_stats.x_num_stats['std_log'].to(
            device=self.device)

        full_mean_values = torch.zeros(self.num_cols, device=self.device)
        full_mean_values[self.col_mask] = mean_log_values if self.apply_power_transform else mean_values

        full_std_values = torch.ones(self.num_cols, device=self.device)
        full_std_values[self.col_mask] = std_log_values if self.apply_power_transform else std_values

        transform_mask = (tok_tensor > self.unk_tok.unsqueeze(1)
                          ) & self.col_mask.unsqueeze(1)

        transformed_tensor = self.power_transform(val_tensor) \
            if self.apply_power_transform else val_tensor

        transformed_tensor = torch.where(
            transform_mask, ((transformed_tensor - full_mean_values.unsqueeze(1)) /
                             (full_std_values.unsqueeze(1) + 1e-8)),
            val_tensor
        )

        self.dataset_x_tok = tok_tensor.transpose(0, 1).contiguous()
        self.dataset_x_val = transformed_tensor.transpose(0, 1).contiguous()

    def split_dataset(self):
        if self.validate_split is None:
            n_validate = 0
        else:
            n_validate = int(self.num_rows * self.validate_split)
        assert self.num_rows > n_validate >= 0
        self.n_validate = n_validate
        self.n_train = self.num_rows - n_validate
        permuted_indices = torch.randperm(
            self.num_rows, device=self.device, generator=self.rng)
        self.train_split_indices = permuted_indices[self.n_validate:]
        self.validate_split_indices = permuted_indices[:self.n_validate]
