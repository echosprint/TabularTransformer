from abc import ABC, ABCMeta, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Union


class ReaderMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        # Wrap the read_data_file method if it exists
        original_read_data_file = dct.get('read_data_file')
        if original_read_data_file:
            def new_read_data_file(self, *args, **kwargs):
                self.pre_read_data()
                result = original_read_data_file(self, *args, **kwargs)
                self.post_read_data(result)
                return result
            dct['read_data_file'] = new_read_data_file
        return super().__new__(cls, name, bases, dct)


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
            assert col in df.columns, f"""ensure_numerical_cols: {
                col} not in data columns"""
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except ValueError as e:
                raise ValueError(
                    f"""Failed to apply `pd.to_numeric` on column [{col}]: {e}""")

        for col in self.ensure_categorical_cols:
            assert col in df.columns, f"""ensure_categorical_cols: {
                col} not in data columns"""
            try:
                df[col] = df[col].astype(str)
            except ValueError as e:
                raise ValueError(
                    f"Failed to cast column [{col}] to string: {e}")
