from typing import Dict, List, Optional
import numpy as np
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import torch

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
