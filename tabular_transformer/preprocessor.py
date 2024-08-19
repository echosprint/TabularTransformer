import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from dataclasses import dataclass
from typing import Dict, Union, Optional
from tabular_transformer.util import FeatureType, CATEGORICAL_UNK, SCALAR_NUMERIC, SCALAR_UNK
from random import Random


@dataclass
class CategoricalStats:
    # categorical variable frequency for all the data
    valid_cats: Dict[str, int]  # {Dog: 500, Cat: 300, Cow: 200}
    minimal_cout: int  # minimal count for a valid category, such like 100


@dataclass
class NumericalStats:
    # scalar variable statistics for all the data points
    min_value: np.number
    max_value: np.number
    mean: np.number
    std: np.number
    logmean: np.number
    logstd: np.number


def power_transform(value):
    """
    a simplified version of power transform
    Yeo-Johnson transform
    x_i = ln(x_i + 1)   if x_i  >= 0 and lambda = 0
    x_i = -ln(-x_i + 1) if x_i < 0 and lambda = 2
    """
    # return -np.log(-value + 1) if value < 0 else np.log(value + 1)
    return -np.log1p(-value) if value < 0 else np.log1p(value)


def inverse_power_transform(value):
    return -np.expm1(-value) if value < 0 else np.expm1(value)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_data(value, mean, std):
    """
    z-score standardization
    z = (x - u) / s
    maybe we should make the normalized value into (0, 2) interval with 2*sigmoid map mean to 1
    """
    value = (value - mean) / (std + 1e-8)
    # value = sigmoid(value) * 2
    # value = sigmoid(value)
    return value


def inverse_normalize_data(value, mean, std):
    return value * (std + 1e-8) + mean


def remove_outliers(value, mean, std, n_sigma=4) -> np.number:
    if np.isnan(value):
        return np.nan
    cut_off = std * n_sigma
    lower, upper = mean - cut_off, mean + cut_off

    if value < -np.log(1+np.abs(value)) + lower:
        return np.nan
    if value > np.log(1+np.abs(value)) + upper:
        return np.nan
    return value


def data_stats(df: pd.DataFrame, min_cat_count: int = 1000):
    """
    Calculate statistics for each column in the DataFrame.
    For categorical columns, count the occurrences of each unique value.
    For numeric columns, calculate the max, min, mean, and standard deviation.
    """
    feature_stats = {}
    feature_type = {}

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            power_col = df[col].map(lambda x: power_transform(x))
            feature_stats[col] = NumericalStats(
                min_value=df[col].min(),
                max_value=df[col].max(),
                mean=df[col].mean(),
                std=df[col].std(),
                logmean=power_col.mean(),
                logstd=power_col.std(),
            )
            feature_type[col] = FeatureType.NUMERICAL
        else:
            value_counts = df[col].value_counts()
            filtered_counts = value_counts[value_counts >=
                                           min_cat_count].to_dict()
            feature_stats[col] = CategoricalStats(
                valid_cats=filtered_counts, minimal_cout=min_cat_count)
            feature_type[col] = FeatureType.CATEGORICAL
    return feature_stats, feature_type


def generate_feature_vocab(feature_stats) -> Dict[str, int]:
    feature_vocab = {}
    i = 0
    for feat, stats in feature_stats.items():
        if isinstance(stats, CategoricalStats):
            feature_vocab[f"{feat.strip()}_{CATEGORICAL_UNK}"] = i
            i += 1
            for cat in stats.valid_cats:
                assert cat != CATEGORICAL_UNK
                feature_vocab[f"{feat.strip()}_{str(cat).strip()}"] = i
                i += 1
        elif isinstance(stats, NumericalStats):
            feature_vocab[f"{feat.strip()}_{SCALAR_UNK}"] = i
            i += 1
            feature_vocab[f"{feat.strip()}_{SCALAR_NUMERIC}"] = i
            i += 1
        else:
            raise ValueError("bad stats")
    return feature_vocab


def random_mark_unk(rng: Random, data: pd.DataFrame,
                    feature_type: Dict[str, FeatureType],
                    unk_ratio: Optional[Dict[str, float]] = None,
                    unk_ratio_default: Optional[float] = None,
                    ):
    if (unk_ratio is None or len(unk_ratio) == 0) and (unk_ratio_default is None or unk_ratio_default <= 0):
        return
    assert unk_ratio_default is not None and 1 > unk_ratio_default > 0
    unk_ratio = {} if unk_ratio is None else unk_ratio

    data_size = len(data)

    for col in data.columns:
        unk_ratio_col = unk_ratio.get(col, unk_ratio_default)
        num_replacements = int(data_size * unk_ratio_col)

        if num_replacements <= 0:
            if data_size < 4:
                raise ValueError("batch size too small for random mark unk")
            else:
                num_replacements = 1
        assert 0 < num_replacements < data_size
        random_indices = rng.sample(
            data.index.to_list(), k=num_replacements)

        if feature_type[col] is FeatureType.CATEGORICAL:
            data.loc[random_indices, col] = CATEGORICAL_UNK
        elif feature_type[col] is FeatureType.NUMERICAL:
            data.loc[random_indices, col] = np.nan
        else:
            raise ValueError("bad featuretype")


def normalize_and_transform(data: pd.DataFrame,
                            feature_type: Dict[str, FeatureType],
                            feature_stats: Dict[str, Union[CategoricalStats, NumericalStats]],
                            apply_power_transform=True,
                            remove_outlier=False):
    for col in data.columns:
        col_type = feature_type[col]
        if col_type is FeatureType.NUMERICAL:
            col_stats: NumericalStats = feature_stats[col]

            if remove_outlier:
                data[col] = data[col].map(lambda x: remove_outliers(
                    x, col_stats.mean, col_stats.std))

            if apply_power_transform:
                data[col] = data[col].map(lambda x: power_transform(x))
                data[col] = data[col].map(lambda x: normalize_data(
                    x, col_stats.logmean, col_stats.logstd))
            else:
                data[col] = data[col].map(lambda x: normalize_data(
                    x, col_stats.mean, col_stats.std))


def preprocess(rng: Random,
               data: pd.DataFrame,
               feature_type: Dict[str, FeatureType],
               feature_stats: Dict[str, Union[CategoricalStats, NumericalStats]],
               apply_power_transform=True,
               remove_outlier=False,
               unk_ratio: Optional[Dict[str, float]] = None,
               unk_ratio_default: Optional[float] = None,
               ) -> pd.DataFrame:
    data = data.copy()
    data.reset_index(drop=True, inplace=True)

    normalize_and_transform(data, feature_type, feature_stats,
                            apply_power_transform, remove_outlier)

    random_mark_unk(rng, data, feature_type, unk_ratio, unk_ratio_default)

    return data
