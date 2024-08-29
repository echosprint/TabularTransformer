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
