from dataclasses import dataclass, replace, field
from typing import Optional, Dict, List, Tuple
from typing_extensions import Literal
import torch


@dataclass
class FeatureStats:
    x_col_type: List[Tuple[str, Literal['cat', 'num']]] = field(default_factory=list)  # noqa: E501
    x_cls_dict: Dict[str, List[str]] = field(default_factory=dict)
    x_num_stats: Dict[Literal['mean', 'std', 'mean_log', 'std_log'], torch.Tensor] = field(default_factory=dict)  # noqa: E501

    y_type: Optional[Literal['cat', 'num']] = None
    y_cls: Optional[List[str]] = None
    y_num_stats: Optional[Tuple[float, float, float, float]] = None

    @property
    def vocab(self):
        vocab = {f"{col}_unk": idx for idx,
                 (col, _) in enumerate(self.x_col_type)}
        cls_num = len(vocab)
        for col, ty in self.x_col_type:
            if ty == 'cat':
                cls = self.x_cls_dict.get(col, None)
                if cls is None:
                    return {}
                vocab.update(
                    {f"{col}_{cl}": cls_num + i for i, cl in enumerate(cls)})
                cls_num += len(cls)
            else:
                vocab[f"{col}_num"] = cls_num
                cls_num += 1
        return vocab

    @property
    def seq_len(self):
        return len(self.x_col_type)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def label_cls_map(self):
        if self.y_type == 'cat':
            assert self.y_cls is not None
            return {i: cls for i, cls in enumerate(self.y_cls)}
        else:
            return {}

    def __call__(self, **kwargs) -> 'FeatureStats':
        return replace(self, **kwargs)

    def merge_original(self, original) -> 'FeatureStats':
        if original is None:
            return self
        assert isinstance(original, FeatureStats)
        assert self.x_col_type == original.x_col_type, "tabular features not compatible."

        if self.y_type is not None and original.y_type is None:
            return original(y_type=self.y_type,
                            y_cls=self.y_cls,
                            y_num_stats=self.y_num_stats)
        assert self.y_type is None or original.y_type is None or self.y_type == original.y_type
        return original

    def reset_y_stats(self) -> 'FeatureStats':
        return self(y_type=None, y_cls=None, y_num_stats=None)

    def __repr__(self):
        x_col_type_str = ",\n        ".join(
            [f"('{k}',    \t'{v}')" for k, v in self.x_col_type])
        x_cls_dict_str = ",\n        ".join(
            [f"'{k}':    \t{v}" for k, v in self.x_cls_dict.items()])
        x_num_stats_str = ",\n        ".join(
            [f"'{k}':  \ttorch.{v}" for k, v in self.x_num_stats.items()])
        vocab_str = ",\n        ".join(
            [f"'{k}':    \t{v}" for k, v in self.vocab.items()])
        return (
            f"FeatureStats(\n"
            f"    x_col_type=[\n        {x_col_type_str}\n    ],\n"
            f"    x_cls_dict={{\n        {x_cls_dict_str}\n    }},\n"
            f"    x_num_stats={{\n        {x_num_stats_str}\n    }},\n"
            f"    y_type={repr(self.y_type)},\n"  # noqa: E501
            f"    y_cls={self.y_cls},\n"
            f"    y_num_stats={[round(elem, 4) for elem in self.y_num_stats] if self.y_num_stats is not None else None},\n"  # noqa: E501
            f")"
        )
