# get start
hello

# Python init

```py
import tabular_transformer as ttf
import pandas as pd
import torch

class IncomeDataReader(ttf.DataReader):
    ensure_categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
    ensure_numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

    def read_data_file(self, file_path):
        df = pd.read_csv(file_path)
        return df
```
