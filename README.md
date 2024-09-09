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

Here we take [Adult Income dataset](https://huggingface.co/datasets/scikit-learn/adult-census-income) as an example to show the usage of `tabular_transformer` package, more examples see the [notebooks](https://github.com/echosprint/TabularTransformer/tree/main/notebooks) folder in this repo.

 <a target="_blank" href="https://colab.research.google.com/github/echosprint/TabularTransformer/blob/main/notebooks/supervised_training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
import tabular_transformer as ttf
import torch

income_dataset_path = ttf.prepare_income_dataset()

categorical_cols = [
    'workclass', 'education',
    'marital.status', 'occupation',
    'relationship', 'race', 'sex',
    'native.country', 'income']

numerical_cols = [
    'age', 'fnlwgt', 'education.num',
    'capital.gain', 'capital.loss',
    'hours.per.week']

income_reader = ttf.DataReader(
    file_path=income_dataset_path,
    ensure_categorical_cols=categorical_cols,
    ensure_numerical_cols=numerical_cols,
    label='income',
)

split = income_reader.split_data({'test': 0.2, 'train': -1})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() \
    and torch.cuda.is_bf16_supported() else 'float16'

ts = ttf.TrainSettings(device=device, dtype=dtype)

tp = ttf.TrainParameters(max_iters=3000, learning_rate=5e-4,
                         output_dim=1, loss_type='BINCE',
                         batch_size=128, eval_interval=100,
                         eval_iters=20, warmup_iters=100,
                         validate_split=0.2)

hp = ttf.HyperParameters(dim=64, n_layers=6)

trainer = ttf.Trainer(hp=hp, ts=ts)

trainer.train(
    data_reader=income_reader(file_path=split['train']),
    tp=tp)

predictor = ttf.Predictor(checkpoint='out/ckpt.pt')

predictor.predict(
    data_reader=income_reader(file_path=split['test']),
    save_as="prediction_income.csv"
)
```
Comparison
----------

We used [Higgs](https://archive.ics.uci.edu/dataset/280/higgs) dataset to conduct our comparison experiment. Details of data are listed in the following tables:

| Training Samples | Features | Test Set Description                 | Task                  |
|------------------|----------|--------------------------------------|-----------------------|
| 10,500,000       | 28       | Last 500,000 samples as the test set | Binary classification |


We computed accuracy metric only on the test data set. check [benchmark source](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#accuracy).
| Data  | Metric | XGBoost | XGBoost_Hist | LightGBM       | TabularTransformer |
|-------|--------|---------|--------------|----------------|--------------------|
| Higgs | AUC    | 0.839593| 0.845314     | 0.845724       | **0.848628**       |

To reproduce the result, please check the [source code](https://github.com/echosprint/TabularTransformer/blob/main/notebooks/higgs_classification.ipynb)

Support
-------

Open **bug reports** and **feature requests** on [GitHub issues](https://github.com/echosprint/TabularTransformer/issues).


Reference Papers
----------------

Xin Huang and Ashish Khetan and Milan Cvitkovic and Zohar Karnin. "[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)". arXiv, 2020.

Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan. "[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)". arXiv, 2020.

Levin, Roman and Cherepanova, Valeriia and Schwarzschild, Avi and Bansal, Arpit and Bruss, C Bayan and Goldstein, Tom and Wilson, Andrew Gordon and Goldblum, Micah. "[Transfer Learning with Deep Tabular Models](https://arxiv.org/abs/2206.15306)". arXiv, 2022.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/echosprint/TabularTransformer/blob/main/LICENSE) for additional details.

Thanks
-------

The prototype of this project is adapted from python parts of [Andrej Karpathy](https://x.com/karpathy)'s [Llama2.c](https://github.com/karpathy/llama2.c), Andrej is a mentor, wish him great success with his startup.