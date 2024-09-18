# TabularTransformer 

## Welcome to TabularTransformer’s documentation!

TabularTransformer is a lightweight, end-to-end deep learning framework built with PyTorch, leveraging the power of the Transformer architecture. It is designed to be scalable and efficient with the following advantages:

- Streamlined workflow with no need for preprocessing or handling missing values.
- Unleashing the power of Transformer on tabular data domain.
- Native GPU support through PyTorch.
- Minimal APIs to get started quickly.
- Capable of handling large-scale data.

## Architecture of TabularTransformer


The model consists of three main parts:

- Embedding Layer: Each column in the tabular data, designated as either `Categorical` or `Numerical`, undergoes a two-part embedding process. Each column scalar value is considered to have two components: a `class` component and a `value` component. The `class` component is embedded using a simple lookup embedding similar to token embedding in `LLM`, while the `value` component is mapped into an n-dim space using absolute position encodings, inspired by  Vaswani et al.'s paper "Attention is All You Need" though with slight modifications. These two embeddings are then combined through addition, forming the embedding of the tabular column feature with the shape of $(n_row, n_col, n_dim)$, preparing the input for further processing by the Transformer.

- Transformer: The core of the model is the Transformer, which consists of multiple layers designed to capture contextual relationships between column features. By processing the embedded features through attention mechanisms, the Transformer is able to model dependencies and interactions between different column features, enriching each feature with contextual information from the others.

- Multi-Layer Perceptron (MLP): Once the Transformer generates the contextual embeddings, they are compressed into a lower-dimensional space and concatenated before being fed into an MLP. Serving as the final component of the model, the MLP aggregates and processes these representations to produce the final output, which may be a regression or classification result, depending on the specific task.

These three components work together to encode tabular data in a way that captures the relationships and interactions between column features, ultimately producing a highly contextual embedding that enhances predictive performance.

![TabularTransformer architecture](assets/arch.svg)
