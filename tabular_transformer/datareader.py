from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import random
import pandas as pd
import os
import torch
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow as pa
import numpy as np
from typing import Dict, List, Optional


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


header: bool = True
ensure_categorical_cols: List[str] = ['city', 'sex']
ensure_numerical_cols: List[str] = ['age', 'weight', 'income']
column_names: Optional[List[str]] = ['city', 'age', 'weight', 'income', 'sex']

file_path = 'test.csv'

apply_power_transform = True

seed = 42
device = 'cuda'
torch.manual_seed(seed)
unk_ratio = 0.2

cat_schema = [(col, pa.string()) for col in ensure_categorical_cols]
num_schema = [(col, pa.float32()) for col in ensure_numerical_cols]
schema = pa.schema(cat_schema + num_schema)

assert header or column_names is not None, \
    "if no header in dataset file, you must denfine `column_names`"

table = csv.read_csv(
    file_path,
    read_options=csv.ReadOptions(
        column_names=column_names if not header else None),
    convert_options=csv.ConvertOptions(column_types=schema)
)

table_col_names = table.column_names

assert column_names is None or column_names == table_col_names, \
    "`column_names` not right."
assert len(set(ensure_categorical_cols) & set(ensure_numerical_cols)) == 0, \
    "column cannot be categorical and numerical"
assert set(ensure_categorical_cols + ensure_numerical_cols) == set(table_col_names), \
    "all columns must be set either categorical or numerical"

min_cat_count = 2

n_col = len(table_col_names)
cls_num = n_col

tok_table = []
val_table = []

cls_dict: Dict[str, List[str]] = {}
col_type: Dict[str, str] = {}

for idx, col in enumerate(table_col_names):
    if col in ensure_categorical_cols:
        col_type[col] = 'cat'
        cls_counts = pc.value_counts(table[col])

        valid_cls_counts = {
            count_struct['values'].as_py(): count_struct['counts'].as_py()
            for count_struct in cls_counts
            if count_struct['values'].is_valid and
            len(str(count_struct['values'])) > 0 and
            count_struct['counts'].as_py() >= min_cat_count
        }

        valid_cls = list(valid_cls_counts.keys())

        assert len(valid_cls) > 0, \
            f"no class in col {col} satisfies `min_cat_count`"

        cls_dict[col] = valid_cls

        int_col = pc.index_in(
            table[col], value_set=pa.array(valid_cls))

        int_col = pc.add(int_col, cls_num)
        int_col = pc.fill_null(int_col, idx)
        int_col = pc.cast(int_col, pa.int16())

        # tok_table = tok_table.set_column(idx, col, int_col)
        tok_table.append(int_col.to_numpy().astype(np.int16))

        scalar1 = pa.scalar(1.0, type=pa.float32())
        # val_table = val_table.set_column(
        #     idx, col,
        #     pa.repeat(scalar1, len(int_col)))
        val_table.append(np.full(len(int_col), 1.0, dtype=np.float32))

        cls_num += len(valid_cls)

    elif col in ensure_numerical_cols:
        col_type[col] = 'num'
        valid_check = pc.is_valid(table[col])
        tok_col = pc.if_else(valid_check, cls_num, idx)
        tok_col = pc.cast(tok_col, pa.int16())
        # tok_table = tok_table.set_column(idx, col, tok_col)
        tok_table.append(tok_col.to_numpy().astype(np.int16))
        fill_col = pc.fill_null(table[col], 1.0)
        # val_table = val_table.set_column(idx, col, fill_col)
        val_table.append(fill_col.to_numpy().astype(np.float32))
        cls_num += 1

    else:
        raise ValueError("Bad column name")

    assert max(cls_num, idx) < 32767

tok_numpy = np.array(tok_table)
val_numpy = np.array(val_table)

tok_tensor = torch.tensor(tok_numpy)
val_tensor = torch.tensor(val_numpy)

col_mask = torch.tensor(
    [False if ty == 'cat' else True
     for _, ty in col_type.items()], device=device
)


tok_tensor = tok_tensor.to(device, non_blocking=True)
val_tensor = val_tensor.to(device, non_blocking=True)


unk_val = torch.arange(n_col, device=tok_tensor.device, dtype=torch.int16)


def power_transform(tensor: torch.Tensor):
    return torch.where(
        tensor < 0, -torch.log1p(-tensor), torch.log1p(tensor))


def cal_mean_std(tok_tensor: torch.Tensor,
                 val_tensor: torch.Tensor,
                 col_mask: torch.Tensor,
                 log_transform: bool = False):

    unk_val = torch.arange(col_mask.size(0),
                           device=tok_tensor.device)

    null_mask = (tok_tensor[col_mask] > unk_val[col_mask].unsqueeze(1)).float()

    val_tensor = val_tensor[col_mask]

    if log_transform:
        val_tensor = power_transform(val_tensor)

    non_null_tensor = val_tensor * null_mask

    mean_values = non_null_tensor.sum(dim=1) / null_mask.sum(dim=1)

    squared_errors = (
        (non_null_tensor - mean_values.unsqueeze(1)) ** 2) * null_mask

    # pandas use `n-1` not `n` as denominator for std calculation
    std_values = torch.sqrt(squared_errors.sum(dim=1) / (null_mask.sum(dim=1)))

    return mean_values, std_values


mean_values, std_values = cal_mean_std(tok_tensor, val_tensor, col_mask)
mean_log_values, std_log_values = cal_mean_std(
    tok_tensor, val_tensor, col_mask, log_transform=True)

full_mean_values = torch.zeros(n_col, device=device)
full_mean_values[col_mask] = mean_log_values if apply_power_transform else mean_values

full_std_values = torch.ones(n_col, device=device)
full_std_values[col_mask] = std_log_values if apply_power_transform else std_values

transform_mask = (tok_tensor > unk_val.unsqueeze(1)) & col_mask.unsqueeze(1)

transformed_tensor = power_transform(val_tensor) \
    if apply_power_transform else val_tensor

transformed_tensor = torch.where(
    transform_mask, ((transformed_tensor - full_mean_values.unsqueeze(1)) /
                     (full_std_values.unsqueeze(1) + 1e-8)),
    val_tensor
)

dataset_tok = tok_tensor.transpose(0, 1).contiguous()
dataset_val = transformed_tensor.transpose(0, 1).contiguous()

rand_unknown_mask = torch.rand(dataset_tok.shape, device=device) >= unk_ratio

dataset_tok_unk = torch.where(rand_unknown_mask,
                              dataset_tok,
                              unk_val.unsqueeze(0).repeat(
                                  dataset_tok.shape[0], 1)
                              )
dataset_val_unk = torch.where(rand_unknown_mask,
                              dataset_val,
                              torch.ones(dataset_val.shape, device=device)
                              )

permuted_indices = torch.randperm(dataset_tok.size(0), device=device)
