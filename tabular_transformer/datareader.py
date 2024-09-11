from pathlib import Path
from typing import Dict, Literal, Optional, Union, List
import pandas as pd
import os
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as parquet
import pyarrow as pa
import numpy as np
import inspect


class DataReader():
    file_path: Path
    ensure_categorical_cols: List[str]
    ensure_numerical_cols: List[str]
    label: Optional[str]
    id: Optional[str]
    header: bool
    column_names: Optional[List[str]]
    file_type: Literal['csv', 'parquet']

    def __init__(self,
                 file_path: Union[str, Path],
                 ensure_categorical_cols: List[str],
                 ensure_numerical_cols: List[str],
                 label: Optional[str],
                 id: Optional[str] = None,
                 header: bool = True,
                 column_names: Optional[List[str]] = None):
        self.file_path = Path(file_path)
        self._check_file_types()

        self.column_names = column_names
        self.label = label
        self.id = id
        assert self.id is None or self.id != self.label
        assert self.id is None or self.id in ensure_categorical_cols, \
            f"id column `{id}` must be `categorical`"

        self.ensure_categorical_cols = ensure_categorical_cols
        self.ensure_numerical_cols = ensure_numerical_cols
        self._check_cat_num_cols()

        assert header or column_names is not None, \
            "if no header in data file, you must denfine `column_names`"
        self.header = header

    def __call__(self, **kwargs):
        sig = inspect.signature(self.__init__)
        paras = [param.name for param in sig.parameters.values()
                 if param.name != 'self']
        assert set(kwargs.keys()).issubset(set(paras)), \
            f"bad arguments: {set(kwargs.keys()) - set(paras)}"

        original_val = {key: getattr(self, key) for key in paras}
        if 'label' in kwargs and kwargs['label'] is None and original_val['label'] is not None:
            label_col = original_val['label']
            original_val['ensure_categorical_cols'] = \
                [col for col in original_val['ensure_categorical_cols']
                    if col != label_col]
            original_val['ensure_numerical_cols'] = \
                [col for col in original_val['ensure_numerical_cols']
                    if col != label_col]
            if original_val['column_names'] is not None:
                original_val['column_names'] = \
                    [col for col in original_val['column_names']
                     if col != label_col]
        original_val.update(kwargs)
        return self.__class__(**original_val)

    def read(self) -> pa.Table:
        cat_schema = [(col, pa.string())
                      for col in self.ensure_categorical_cols]
        num_schema = [(col, pa.float32())
                      for col in self.ensure_numerical_cols]
        schema = pa.schema(cat_schema + num_schema)

        print('start reading file, it may take a while..')
        if self.file_type == 'csv':
            table = csv.read_csv(
                self.file_path,
                read_options=csv.ReadOptions(
                    column_names=self.column_names if not self.header else None),
                convert_options=csv.ConvertOptions(column_types=schema)
            )
        else:
            table = parquet.read_table(self.file_path)
            reordered_schema = pa.schema(
                [schema.field(col) for col in table.column_names])
            table = table.cast(reordered_schema)
        print('read file completed.')

        table_col_names = table.column_names
        print(f"dataset column names: {table_col_names}")

        assert self.label is None or self.label in table_col_names, \
            f"`label` '{self.label}' not exists in table column names."

        assert self.column_names is None or self.column_names == table_col_names, \
            f"`column_names` not right. Mismatched columns: \
                {set(self.column_names) ^ set(table_col_names)}"

        self.column_names = table_col_names if self.column_names is None else self.column_names

        assert set(self.ensure_categorical_cols).issubset(set(table_col_names)), \
            f"cols specified in `ensure_categorical_cols` not exist in column_names: \
            {set(self.ensure_categorical_cols) - set(table_col_names)}"

        assert set(self.ensure_numerical_cols).issubset(set(table_col_names)), \
            f"cols specified in `ensure_numerical_cols` not exist in column_names: \
            {set(self.ensure_numerical_cols) - set(table_col_names)}"

        assert set(self.ensure_categorical_cols + self.ensure_numerical_cols) == set(table_col_names), \
            f"all columns must be set either in `ensure_categorical_cols` or `ensure_numerical_cols`, missing cols: \
               {set(table_col_names) - set(self.ensure_categorical_cols + self.ensure_numerical_cols)}"

        assert self.id is None or self.id in self.column_names, \
            f"id column `{self.id}` not exists."

        return table

    def _check_file_types(self):
        if self.file_path.suffix == '.csv' or self.file_path.suffixes == ['.csv', '.gz']:
            self.file_type = 'csv'
        elif self.file_path.suffix == '.parquet':
            self.file_type = 'parquet'
        else:
            raise ValueError(
                "DataReader only support file type with extension: `csv`, `csv.gz`, `parquet`")

    def _check_cat_num_cols(self):
        assert isinstance(self.ensure_numerical_cols, list) \
            and (len(self.ensure_numerical_cols) == 0
                 or all(isinstance(e, str) and len(e.strip()) > 0
                        for e in self.ensure_numerical_cols)), \
            "`ensure_numerical_cols` must be list of column names"

        assert isinstance(self.ensure_categorical_cols, list) \
            and (len(self.ensure_categorical_cols) == 0
                 or all(isinstance(e, str) and len(e.strip()) > 0
                        for e in self.ensure_categorical_cols)), \
            "`ensure_categorical_cols` must be list of column names"

        numerical_set = set(self.ensure_numerical_cols)
        categorical_set = set(self.ensure_categorical_cols)
        common_set = numerical_set.intersection(categorical_set)
        assert len(common_set) == 0, \
            f"""{list(common_set)}
                      both in the ensure_numerical_cols and ensure_categorical_cols"""
        if self.column_names is not None:
            assert set(self.ensure_categorical_cols).issubset(set(self.column_names)), \
                f"cols specified in `ensure_categorical_cols` not exist in column_names: \
            {set(self.ensure_categorical_cols) - set(self.column_names)}"

            assert set(self.ensure_numerical_cols).issubset(set(self.column_names)), \
                f"cols specified in `ensure_numerical_cols` not exist in column_names: \
            {set(self.ensure_numerical_cols) - set(self.column_names)}"

    def split_data(self, split: Dict[str, float | int],
                   seed: Optional[int] = 1337,
                   override: bool = True,
                   save_as: Literal['csv', 'csv.gz', 'parquet'] = 'csv') -> Dict[str, Path]:

        assert isinstance(split, dict), "`split` must be Dict[str, float|int]"
        assert save_as in ['csv', 'csv.gz', 'parquet']

        file_path: Path = self.file_path
        base_stem = file_path.stem.split('.')[0]
        suffix = f".{save_as}"
        if all(file_path.with_name(f"{base_stem}_{sp}{suffix}").exists()
               for sp in split.keys()) \
                and not override:
            print("splits already exists, skip split.")
            return {f'{sp}': file_path.with_name(f"{base_stem}_{sp}{suffix}")
                    for sp in split.keys()}

        table = self.read()

        data_size = table.num_rows
        ixs = np.arange(data_size)

        if seed is not None:
            rng = np.random.default_rng(seed)
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
            data_part = table.take(ixs[start: end])
            print(f'split: {sp}, n_samples: {part_len}')

            part_path = file_path.with_name(f"{base_stem}_{sp}{suffix}")

            if part_path.exists() and override:
                os.remove(part_path)
                print(f"{part_path} *exists*, delete old split `{sp}`")

            if not part_path.exists():
                print(f"save split `{sp}` at path: {part_path}")

                if save_as == 'csv':
                    csv.write_csv(data_part, part_path)
                elif save_as == 'csv.gz':
                    with pa.output_stream(part_path, compression='gzip') as stream:
                        csv.write_csv(data_part, stream)
                elif save_as == 'parquet':
                    parquet.write_table(data_part, part_path)
                else:
                    raise ValueError("bad file type.")
            else:
                print(f"{part_path} *exists*, skip split `{sp}`")

            fpath[f'{sp}'] = part_path
            start = end
        return fpath

    def __repr__(self):
        return (
            f"DataReader(\n"
            f"  file_path = '{self.file_path}',\n"
            f"  ensure_categorical_cols = {self.ensure_categorical_cols},\n"
            f"  ensure_numerical_cols = {self.ensure_numerical_cols},\n"
            f"  label = {repr(self.label)},\n"
            f"  id = {repr(self.id)},\n"
            f"  header = {self.header},\n"
            f"  column_names = {self.column_names}\n"
            f")"
        )
