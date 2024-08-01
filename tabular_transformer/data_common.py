from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Union


class DataReader(ABC):
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
