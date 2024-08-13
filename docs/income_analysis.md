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

    /home/work/TabularTransformer/data/income/income.csv already exists, skipping download.
    more details see website: https://huggingface.co/datasets/scikit-learn/adult-census-income



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>?</td>
      <td>77053</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>132870</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>?</td>
      <td>186061</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
split = income_reader.split_data({'test': 0.2, 'train': -1})
print(split)
```

    split: test, n_samples: 6512
    /home/work/TabularTransformer/data/income/income_test.csv *already exists*, skip save split `test`
    split: train, n_samples: 26049
    /home/work/TabularTransformer/data/income/income_train.csv *already exists*, skip save split `train`
    {'test': PosixPath('/home/work/TabularTransformer/data/income/income_test.csv'), 'train': PosixPath('/home/work/TabularTransformer/data/income/income_train.csv')}



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

    load dataset from file: /home/work/TabularTransformer/data/income/income_train.csv
    num parameter tensors: 62, with 356,464 parameters
    Transformer num decayed parameter tensors: 43, with 323,648 parameters
    Transformer num non-decayed parameter tensors: 13, with 832 parameters
    Output num decayed parameter tensors: 5, with 31,872 parameters
    Output num non-decayed parameter tensors: 1, with 112 parameters
    using fused AdamW: True
    step 0: train loss 0.6931, val loss 0.6931
    0 | loss 0.6932 | lr 0.000000e+00 | 364.76ms | mfu -100.00%
    1 | loss 0.6930 | lr 5.000000e-06 | 15.06ms | mfu -100.00%
    2 | loss 0.6931 | lr 1.000000e-05 | 15.08ms | mfu -100.00%
    3 | loss 0.6930 | lr 1.500000e-05 | 13.94ms | mfu -100.00%
    ...
    2426 | loss 0.2679 | lr 3.635941e-09 | 13.20ms | mfu  0.09%
    2427 | loss 0.2499 | lr 2.045219e-09 | 13.29ms | mfu  0.09%
    2428 | loss 0.2828 | lr 9.089869e-10 | 13.27ms | mfu  0.09%
    2429 | loss 0.2366 | lr 2.272468e-10 | 13.72ms | mfu  0.09%



```python
predictor = ttf.Predictor(checkpoint='out/ckpt.pt')
predictor.predict(data_reader=IncomeDataReader(split['test']),
                  save_as="prediction_output.csv")
```

    binary cross entropy loss: 0.2770
    auc score: 0.9298
    samples: 6512, accuracy: 0.87
    save prediction output to file: out/prediction_output.csv



```python

```
