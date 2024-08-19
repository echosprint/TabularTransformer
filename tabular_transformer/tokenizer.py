from torch import Tensor
import torch
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from tabular_transformer.util import FeatureType, SCALAR_NUMERIC, SCALAR_UNK, CATEGORICAL_UNK


class Tokenizer():

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
                raise ValueError("column type can be CATEGORICAL or NUMERICAL")

        return torch.stack(feature_tokens_list, dim=1), torch.stack(feature_weight_list, dim=1)

    def __eq__(self, other):
        if isinstance(other, Tokenizer):
            return self.feature_vocab == other.feature_vocab and self.feature_type == other.feature_type
        return False
