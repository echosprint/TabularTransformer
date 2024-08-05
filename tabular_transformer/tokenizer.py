from torch import Tensor
import torch
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from .util import FeatureType, SCALAR_NUMERIC, SCALAR_UNK, CATEGORICAL_UNK


class Tokenizer():

    """
    sample tabular data, [UNK] means the value is missing or anomaly, we treated as UNKNOWN
    +---------+-------+---------------+---------------+
    | Name    | Age   | City          | Income        |
    +---------+-------+---------------+---------------+
    | Alice   | 30    | [UNK]         | 500,000       |
    | Bob     |[UNK]  | San Francisco | 1,000,000     |
    | [UNK]   | 35    | Los Angeles   | 200,000       |
    | Charlie | 35    | Los Angeles   | [UNK]         |
    +---------+-------+---------------+---------------+

    1. Every feature(column) must be one of the two types:
        a) Scalar variables  
            such as height, weight, the number of floors in a building
        b) Categorical variables
            such as race, gender, place of birth, color, type of medication

    2. do some statics to calcuate the feature size, for less frequent categorical value, replace it with [UNK]

        the feature size for the tabular is 11

        FEATURE VOCAB DICTIONARY
        +--------------+-------+
        | feature      | index |
        +--------------+-------+
        | Name_[UNK]   | 0     |
        | Name_Alice   | 1     |
        | Name_Bob     | 2     |
        | Name_Charlie | 3     |
        | Age_[UNK]    | 4     |
        | Age_num      | 5     |
        | City_[UNK]   | 6     |
        | City_SF      | 7     |
        | City_LA      | 8     |
        | Income_[UNK] | 9     |
        | Income_num   | 10    |
        +--------------+-------+

    3. look up the feature dictionary for feature tokens, datatype is integer
        feature_tokens for next step EMBEDDING

                    FEATURE TOKENS
    +---------+-------+---------------+---------------+
    | Name    | Age   | City          | Income        |
    +---------+-------+---------------+---------------+
    | 1       | 5     | 6             | 10            |
    | 2       | 4     | 7             | 10            |
    | 0       | 5     | 8             | 10            |
    | 3       | 5     | 8             | 9             |
    +---------+-------+---------------+---------------+

    4. fill in the feature weights, datatype is float
       the Categorical variable fill 1.0
       the Scalar variable(except [UNK]) fill normalized value

                    FEATURE WEIGHTS
    +---------+-------+---------------+---------------+
    | Name    | Age   | City          | Income        |
    +---------+-------+---------------+---------------+
    | 1.0     |-1.1547| 1.0           | -0.1650       |
    | 1.0     |1.0    | 1.0           | 1.0722        |
    | 1.0     |0.5774 | 1.0           | -0.9073       |
    | 1.0     |0.5774 | 1.0           | 1.0           |
    +---------+-------+---------------+---------------+


    """

    feature_vocab: Dict[str, int]
    feature_type: Dict[str, FeatureType]

    def __init__(
        self,
        feature_vocab: Dict[str, int],
        feature_type: Dict[str, FeatureType],
    ) -> None:

        self.feature_vocab = feature_vocab
        self.feature_type = feature_type

    @property
    def feature_vocab_size(self) -> int:
        return len(self.feature_vocab)

    @property
    def feature_vocab_item(self) -> List[str]:
        return list(self.feature_vocab.keys())

    def encode(self, tab: pd.DataFrame) -> Tuple[Tensor, Tensor]:

        feature_tokens_list: List[Tensor] = []
        feature_weight_list: List[Tensor] = []

        def map_categorical(col, cat):
            key = f"{col.strip()}_{str(cat).strip()}"
            key_unk = f"{col.strip()}_{CATEGORICAL_UNK}"
            return self.feature_vocab[key] if key in self.feature_vocab else self.feature_vocab[key_unk]

        def map_scalar(col, sca):
            return self.feature_vocab[f"{col.strip()}_{SCALAR_UNK}"] if np.isnan(sca) else self.feature_vocab[f"{col.strip()}_{SCALAR_NUMERIC}"]

        for col in tab.columns:
            assert col in self.feature_type
            col_type = self.feature_type[col]
            if col_type is FeatureType.CATEGORICAL:
                t = tab[col].map(lambda x: map_categorical(col, x))
                t1 = torch.tensor(t.to_numpy(), dtype=torch.long)
                t2 = torch.ones_like(t1, dtype=torch.float32)
                feature_tokens_list.append(t1)
                feature_weight_list.append(t2)
            elif col_type is FeatureType.NUMERICAL:
                t = tab[col].map(lambda x: map_scalar(col, x))
                v = tab[col].map(lambda x: 1.0 if np.isnan(x) else x)
                t1 = torch.tensor(t.to_numpy(), dtype=torch.long)
                v1 = torch.tensor(v.to_numpy(), dtype=torch.float32)
                feature_tokens_list.append(t1)
                feature_weight_list.append(v1)
            else:
                raise ValueError("column type can be CATEGORICAL or SCALAR")

        return torch.stack(feature_tokens_list, dim=1), torch.stack(feature_weight_list, dim=1)

    def __eq__(self, other):
        if isinstance(other, Tokenizer):
            return self.feature_vocab == other.feature_vocab and self.feature_type == other.feature_type
        return False
