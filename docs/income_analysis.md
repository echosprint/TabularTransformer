<a target="_blank" href="https://colab.research.google.com/github/echosprint/TabularTransformer/blob/main/income_analysis.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---


**for more details about the [TabularTransformer](https://github.com/echosprint/TabularTransformer) model**,
ckeck the online **[Documents](https://echosprint.github.io/TabularTransformer/)**

---

- This notebook provides a usage example of the
  [TabularTransformer](https://github.com/echosprint/TabularTransformer)
  package.
- Hyperparameters are not tuned and may be suboptimal.


```python
%pip install git+https://github.com/echosprint/TabularTransformer.git
```


```python
import tabular_transformer as ttf
import pandas as pd
import torch
```


```python
income_dataset_path = ttf.prepare_income_dataset()
```


```python

class IncomeDataReader(ttf.DataReader):
    ensure_categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
    ensure_numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

    def read_data_file(self, file_path):
        df = pd.read_csv(file_path)
        return df
```


```python
income_reader = IncomeDataReader(income_dataset_path)
df = income_reader.read_data_file()
df.head(3)
```


```python
split = income_reader.split_data({'test': 0.2, 'train': -1})
print(split)
```


```python
# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() \
            and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'

ts = ttf.TrainSettings(wandb_log=False, 
                       device=device, 
                       dtype=dtype, 
                       eval_iters=20)

tp = ttf.TrainParameters(train_epochs=15,
                         batch_size=128,
                         eval_interval=100,
                         warmup_iters=100,
                         validate_split=0.2)

hp = ttf.HyperParameters(dim=64,
                         n_layers=6)

trainer = ttf.Trainer(hp=hp, ts=ts)
trainer.train(data_reader=IncomeDataReader(split['train']), tp=tp, resume=False)

```


```python
predictor = ttf.Predictor(checkpoint='out/ckpt.pt')
predictor.predict(data_reader=IncomeDataReader(split['test']),
                  save_as="prediction_output.csv")
```
