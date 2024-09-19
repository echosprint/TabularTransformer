# Embedding Pipeline
**Unleashing the power of Transformers on tabular data: Embedding is all you need.**

Effective embedding is crucial for enhancing transformer performance on tabular data. Transformers often underperform in this domain due to the lack of well-constructed embeddings for tabular features.

This document outlines the embedding pipeline of TabularTransformer. It first classifies features as numerical or categorical, then calculates the overall feature vocabulary. Next, it details feature-to-token mapping and normalization of numerical features. Finally, it combines feature token embedding and feature value embeddings by addition as input for the Transformer, offering a clear view of the embedding process.

## Sample Tabular Data
In the sample table below, `[UNK]` denotes missing or anomalous values.

| Occupation | Age   | City          | Income    |
|------------|-------|---------------|-----------|
| Engineer   | 30    | [UNK]         | 500,000   |
| Doctor     | [UNK] | San Francisco | 1,000,000 |
| [UNK]      | 35    | Los Angeles   | 200,000   |
| Artist     | 35    | Los Angeles   | [UNK]     |

## Feature Types

Each feature (column) must be classified as one of the following types:

- **Numerical**: continuous values (e.g., 'age', 'income')
- **Categorical**: discrete categories (e.g., 'gender', 'color')

This classification is based on the semantic meaning of the columns rather than
their stored data types and is determined by your understanding of the data.

## Calculating Feature Vocabulary


### Feature Vocabulary
| Feature              | Index |
|----------------------|-------|
| Occupation_[UNK]     | 0     |
| Age_[UNK]            | 1     |
| City_[UNK]           | 2     |
| Income_[UNK]         | 3     |
| Occupation_Engineer  | 4     |
| Occupation_Doctor    | 5     |
| Occupation_Artist    | 6     |
| Age_[num]            | 7     |
| City_SF              | 8     |
| City_LA              | 9     |
| Income_[num]         | 10    |

## Feature Tokens Lookup
Next, we perform a lookup for feature tokens using the Feature Vocabulary. 

These tokens, represented as integers, are then used in the embedding process of `nn.Embedding` module.

### Feature Tokens Table
| Occupation | Age | City | Income |
|------------|-----|------|--------|
| 4          | 7   | 2    | 10     |
| 5          | 1   | 8    | 10     |
| 0          | 7   | 9    | 10     |
| 6          | 7   | 9    | 3      |

## Feature Values
- **Categorical features**: Assign a value of `1.0`.
- **Numerical features**: Assign their z-score normalized values.
- **UNK**: Assign a value of `1.0`.


if `apply_power_transform` is `True`, apply the transform before normalization.

```python
def power_transform(value):
    return -np.log1p(-value) if value < 0 else np.log1p(value)
```


### Feature Values Table
| Occupation | Age    | City | Income  |
|------------|--------|------|---------|
| 1.0        | -1.1547| 1.0  | -0.1650 |
| 1.0        | 1.0    | 1.0  | 1.0722  |
| 1.0        | 0.5774 | 1.0  | -0.9073 |
| 1.0        | 0.5774 | 1.0  | 1.0     |


Feature Value Embedding is achieved by mapping the normalized scalar value 
to an `n_dim` vector using a method similar to [Absolute Position Encoding](https://arxiv.org/abs/1706.03762).

$$
FVE_{(val, 2i)} = \sin\left(val \times 10000^{\frac{2i}{n\_dim}}\right)
$$

$$
FVE_{(val, 2i+1)} = \cos\left(val \times 10000^{\frac{2i}{n\_dim}}\right)
$$

## Embedding Combination

After the two-step embedding of Feature Token and Feature Value, we obtain two tensors with the shape $(n_{row}, n_{col}, n_{dim})$. We then combine these tensors by adding them together to serve as the input to the transformer.
