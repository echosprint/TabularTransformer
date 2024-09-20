# Embedding Pipeline
**Unleashing the power of Transformers on tabular data: Embedding is all you need.**

Transformers have revolutionized natural language processing by enabling models to capture complex relationships in sequential data. However, applying Transformers directly to tabular data has been challenging due to the unique characteristics of tabular datasets, such as heterogeneous feature types and missing values. One of the primary hurdles is the lack of well-constructed embeddings for tabular features.

In this document, we present a comprehensive embedding pipeline designed to enhance Transformer performance on tabular data by effectively representing both categorical and numerical features.

## Sample Tabular Data

Consider the following sample dataset, where [UNK] denotes missing or anomalous values:

| Occupation | Age   | City          | Income    |
|------------|-------|---------------|-----------|
| Engineer   | 30    | [UNK]         | 500,000   |
| Doctor     | [UNK] | San Francisco | 1,000,000 |
| [UNK]      | 35    | Los Angeles   | 200,000   |
| Artist     | 35    | Los Angeles   | [UNK]     |

This dataset includes both categorical features (Occupation, City) and numerical features (Age, Income), as well as missing values.

## Feature Types

Each feature (column) in the dataset must be classified as one of the following types based on its semantic meaning:

- **Numerical**: Features representing continuous values (e.g., Age, Income).
- **Categorical**: Features representing discrete categories (e.g., Occupation, City).

Whether a feature is considered `Numerical` or `Categorical` depends on your understanding and interpretation of the data.

This classification is essential because it determines how each feature will be processed during embedding.


## Building the Feature Vocabulary

To effectively embed the features, we first construct a **Feature Vocabulary** that maps each unique token in the tabular to a unique index. This vocabulary includes special tokens for missing values (`[UNK]`) and numerical values (`[num]`).

### Feature Vocabulary Table

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
| City_San Francisco   | 8     |
| City_Los Angeles     | 9     |
| Income_[num]         | 10    |

This vocabulary ensures that each unique feature token is associated with a unique index, which will be used in the embedding process.

## Feature Tokens Lookup

Using the Feature Vocabulary, we convert the original tabular data into a table of **Feature Tokens**, where each token is represented by its corresponding index.


### Feature Tokens Table
| Occupation | Age | City | Income |
|------------|-----|------|--------|
| 4          | 7   | 2    | 10     |
| 5          | 1   | 8    | 10     |
| 0          | 7   | 9    | 10     |
| 6          | 7   | 9    | 3      |

These indices will be used with an embedding layer (e.g., PyTorch's `nn.Embedding`) to obtain vector representations of the feature tokens.

## Computing Feature Values

For each feature, we also compute a **Feature Value**, which captures the magnitude of the feature.

### Assigning Values to Missing Values

- **Missing or Unknown Values (`[UNK]`)**: Assign a default value of `1.0`.

### Assigning Values to Features

- **Categorical Features**: Since they represent discrete categories, we assign a default value of `1.0` to all categorical features.

- **Numerical Features**:

  - **Power Transform (Optional)**: If `apply_power_transform` is set to `True`, a power transform is applied to make the data more Gaussian-like before normalization. This can help stabilize variance and minimize skewness.

  - **Normalization**: Numerical features are normalized using z-score normalization to ensure they have zero mean and unit variance.


```python
def power_transform(value):
    return -np.log1p(-value) if value < 0 else np.log1p(value)
```


### Feature Values Table

After applying normalization (and optional power transform), we obtain the following Feature Values:

| Occupation | Age    | City | Income  |
|------------|--------|------|---------|
| 1.0        | -1.1547| 1.0  | -0.1650 |
| 1.0        | 1.0    | 1.0  | 1.0722  |
| 1.0        | 0.5774 | 1.0  | -0.9073 |
| 1.0        | 0.5774 | 1.0  | 1.0     |

Note: The numerical values here are illustrative; in practice, you would compute the exact z-scores.

### Feature Values Embedding

To represent the Feature Values as embeddings, we map each normalized scalar value to a high-dimensional vector using a method similar to [Absolute Position Encoding](https://arxiv.org/abs/1706.03762) used in Transformers.

For each value $val$, and for dimensions $i$ in $0$ to $\frac{n_{dim}}{2}​​−1$, we compute:

$$
FVE_{(val, 2i)} = \sin\left(val \times 10000^{\frac{2i}{n_{dim}}}\right)
$$

$$
FVE_{(val, 2i+1)} = \cos\left(val \times 10000^{\frac{2i}{n_{dim}}}\right)
$$

This results in a vector of size $n_{dim}$ that encodes the scalar value in a way that the model can understand.

## Combining Embeddings

After obtaining the embeddings for Feature Tokens and Feature Values, we have two tensors:
- **Feature Token Embeddings**: Shape $(n_{row}, n_{col}, n_{dim})$
- **Feature Value Embeddings**: Shape $(n_{row}, n_{col}, n_{dim})$

We combine these embeddings by element-wise addition to form the final input tensor to the Transformer:

$$
{Input}_{Transformer​}=Feature Token Embeddings + Feature Value Embeddings
$$

This combined embedding incorporates both the identity of the feature (through the token embedding) and its value (through the value embedding), enabling the Transformer to effectively process tabular data.

