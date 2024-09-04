import tabular_transformer as ttf
import pandas as pd
import torch

income_dataset_path = ttf.prepare_income_dataset()

ensure_categorical_cols = [
    'workclass', 'education',
    'marital.status', 'occupation',
    'relationship', 'race', 'sex',
    'native.country', 'income']

ensure_numerical_cols = [
    'age', 'fnlwgt', 'education.num',
    'capital.gain', 'capital.loss',
    'hours.per.week']


pretrain_ensure_categorical_cols = [
    'pretext_target'
    if x == 'income' else x
    for x in ensure_categorical_cols]


def replace_cols(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['income'], inplace=True)
    df['pretext_target'] = df['occupation']
    df.to_csv(file_path, index=False)


income_reader = ttf.DataReader(
    file_path=income_dataset_path,
    ensure_categorical_cols=ensure_categorical_cols,
    ensure_numerical_cols=ensure_numerical_cols,
    label='income',
)
df = income_reader.read().to_pandas()
print(df.head(3))

split = income_reader.split_data(
    {'pretrain': 0.8, 'finetune': 64, 'ssl_test': -1})
print(split)

replace_cols(split['pretrain'])

pretrain_reader = ttf.DataReader(
    file_path=split['pretrain'],
    ensure_categorical_cols=pretrain_ensure_categorical_cols,
    ensure_numerical_cols=ensure_numerical_cols,
    label="pretext_target")

pdf = pretrain_reader.read().to_pandas()
print(pdf.head(3))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() \
    and torch.cuda.is_bf16_supported() else 'float16'

ts = ttf.TrainSettings(wandb_log=False,
                       device=device,
                       dtype=dtype,
                       )


hp = ttf.HyperParameters(dim=64,
                         n_layers=6)

trainer = ttf.Trainer(hp=hp, ts=ts)

pretrain_tp = ttf.TrainParameters(
    max_iters=20,
    loss_type='SUPCON',
    batch_size=128,
    output_dim=16,
    unk_ratio={'occupation': 0.50},
    eval_interval=20,
    eval_iters=10,
    warmup_iters=1,
    validate_split=0.2,
    always_save_checkpoint=True,
    output_checkpoint='pretrain_ckpt.pt')

trainer.train(
    data_reader=pretrain_reader,
    tp=pretrain_tp,
    resume=False)


finetune_tp = ttf.TrainParameters(
    transformer_lr=5e-6,
    output_head_lr=5e-5,
    lr_scheduler='constant',
    max_iters=50,
    loss_type='BINCE',
    batch_size=64,
    output_dim=1,
    eval_interval=10,
    always_save_checkpoint=True,
    eval_iters=1,
    # warmup_iters=10,
    warmup_iters=1,
    validate_split=0.0,
    input_checkpoint='pretrain_ckpt.pt',
    output_checkpoint='finetune_ckpt.pt',
)

trainer.train(
    data_reader=income_reader(file_path=split['finetune']),
    tp=finetune_tp,
    resume=True,
    replace_output_head=True)

predictor = ttf.Predictor(checkpoint='out/finetune_ckpt.pt')
predictor.predict(data_reader=income_reader(file_path=split['ssl_test']),
                  save_as="prediction_output.csv")

# ----------------------------------------------------------------------------

fish_dataset_path = ttf.prepare_fish_dataset()

ensure_categorical_cols = ['Species']
ensure_numerical_cols = ['Weight', 'Length1',
                         'Length2', 'Length3',
                         'Height', 'Width']


fish_data_reader = ttf.DataReader(
    file_path='./data/fish/fish.csv',
    ensure_categorical_cols=ensure_categorical_cols,
    ensure_numerical_cols=ensure_numerical_cols,
    label='Width',
    header=True,
    column_names=None)

df = fish_data_reader.read().to_pandas()
print(df.head(3))

split = fish_data_reader.split_data(
    {'train': 0.8, 'test': -1},
    seed=42,
    save_as='csv')

print(split)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() \
    and torch.cuda.is_bf16_supported() else 'float16'


ts = ttf.TrainSettings(wandb_log=False,
                       device=device,
                       dtype=dtype,
                       )


hp = ttf.HyperParameters(dim=32,
                         n_layers=4,
                         n_heads=4,
                         output_forward_dim=4,
                         output_hidden_dim=64)

trainer = ttf.Trainer(hp=hp, ts=ts)

train_tp = ttf.TrainParameters(
    learning_rate=5e-4,
    lr_scheduler='constant',
    max_iters=30,
    loss_type='MSE',
    batch_size=16,
    output_dim=1,
    eval_interval=10,
    eval_iters=2,
    warmup_iters=5,
    validate_split=0.23,
    output_checkpoint='fish_ckpt.pt',
    input_checkpoint='fish_ckpt.pt')

trainer.train(
    data_reader=fish_data_reader(file_path=split['train']),
    tp=train_tp,
    resume=False)

predictor = ttf.Predictor(checkpoint='out/fish_ckpt.pt')
predictor.predict(data_reader=fish_data_reader(file_path=split['test']),
                  save_as="fish_predictions.csv")
