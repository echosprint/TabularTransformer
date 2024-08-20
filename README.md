Transformer adapted for tabular data domain
===============================


TabularTransformer is a lightweight, end-to-end deep learning framework built with PyTorch, leveraging the power of the Transformer architecture. It is designed to be scalable and efficient with the following advantages:

- Streamlined workflow with no need for preprocessing or handling missing values.
- Unleashing the power of Transformer on tabular data domain.
- Native GPU support through PyTorch.
- Minimal APIs to get started quickly.
- Capable of handling large-scale data.


Get Started and Documentation
-----------------------------

Our primary documentation is at https://echosprint.github.io/TabularTransformer/ and is generated from this repository. 

### Installation:

```bash
$ pip install tabular-transformer
```

### Usage

here we use [Adult Income dataset](https://huggingface.co/datasets/scikit-learn/adult-census-income) as an example to show the usage of `tabular_transformer` package 

 <a target="_blank" href="https://colab.research.google.com/github/echosprint/TabularTransformer/blob/main/notebooks/supervised_training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
import tabular_transformer as ttf
import pandas as pd
import torch

# download the dataset
income_dataset_path = ttf.prepare_income_dataset()

class IncomeDataReader(ttf.DataReader):
    # make sure interpret columns correctly 
    ensure_categorical_cols = []
    ensure_numerical_cols = []

    # load data 
    def read_data_file(self, file_path):
        df = pd.read_csv(file_path)
        return df

income_reader = IncomeDataReader(income_dataset_path)

# split 20% as `test`, rest of as `train`
split = income_reader.split_data({'test': 0.2, 'train': -1})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() \
    and torch.cuda.is_bf16_supported() else 'float16'

ts = ttf.TrainSettings(device=device, dtype=dtype)

tp = ttf.TrainParameters(train_epochs=15, learning_rate=5e-4,
                         batch_size=128, eval_interval=100,
                         eval_iters=20, warmup_iters=100,
                         validate_split=0.2)

hp = ttf.HyperParameters(dim=64, n_layers=6)

# use `HyperParameters` and `TrainSettings` to construct `Trainer`
trainer = ttf.Trainer(hp=hp, ts=ts)
# use split `train` to train with `TrainParameters`
trainer.train(data_reader=IncomeDataReader(split['train']), tp=tp)

# load Pytorch checkpoint
predictor = ttf.Predictor(checkpoint='out/ckpt.pt')
# use split `test` to predict
predictor.predict(
    data_reader=IncomeDataReader(split['test']),
    save_as="prediction_income.csv"
)
```

Support
-------

Open **bug reports** and **feature requests** on [GitHub issues](https://github.com/echosprint/TabularTransformer/issues).


Reference Papers
----------------

Xin Huang and Ashish Khetan and Milan Cvitkovic and Zohar Karnin. "[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)". arXiv, 2020.

Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan. "[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)". arXiv, 2020.


License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/echosprint/TabularTransformer/blob/main/LICENSE) for additional details.