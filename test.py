import tabular_transformer as ttf
import pandas as pd
import torch

income_dataset_path = ttf.prepare_income_dataset()


class IncomeDataReader(ttf.DataReader):
    ensure_categorical_cols = [
        'workclass', 'education',
        'marital.status', 'occupation',
        'relationship', 'race', 'sex',
        'native.country', 'income']

    ensure_numerical_cols = [
        'age', 'fnlwgt', 'education.num',
        'capital.gain', 'capital.loss',
        'hours.per.week']

    def read_data_file(self, file_path):
        df = pd.read_csv(file_path)
        return df


class PretrainIncomeDataReader(IncomeDataReader):
    ensure_categorical_cols = [
        'pretext_target'
        if x == 'income' else x
        for x in IncomeDataReader.ensure_categorical_cols]

    def read_data_file(self, file_path):
        df = pd.read_csv(file_path)
        df.drop(columns=['income'], inplace=True)
        df['pretext_target'] = df['occupation']
        return df


income_reader = IncomeDataReader(income_dataset_path)
df = income_reader.read_data_file()
df.head(3)

split = income_reader.split_data(
    {'pretrain': 0.8, 'finetune': 64, 'ssl_test': -1})
print(split)

pretrain_reader = PretrainIncomeDataReader(split['pretrain'])
pdf = pretrain_reader.read_data_file()
pdf.head(3)

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
    # train_epochs=30,
    train_epochs=1,
    loss_type='SUPCON',
    batch_size=128,
    output_dim=16,
    unk_ratio={'occupation': 0.50},
    eval_interval=100,
    eval_iters=20,
    warmup_iters=1,
    # warmup_iters=500,
    validate_split=0.2,
    always_save_checkpoint=True,
    output_checkpoint='pretrain_ckpt.pt')

trainer.train(
    data_reader=pretrain_reader,
    tp=pretrain_tp,
    resume=False)


finetune_tp = ttf.TrainParameters(
    # transformer_lr=0.0,
    transformer_lr=5e-6,
    output_head_lr=5e-5,
    lr_scheduler='constant',
    # train_epochs=250,
    train_epochs=5,
    loss_type='BINCE',
    batch_size=64,
    output_dim=1,
    eval_interval=4,
    always_save_checkpoint=True,
    eval_iters=1,
    # warmup_iters=10,
    warmup_iters=1,
    validate_split=0.0,
    input_checkpoint='pretrain_ckpt.pt',
    output_checkpoint='finetune_ckpt.pt',
)

trainer.train(
    data_reader=IncomeDataReader(split['finetune']),
    tp=finetune_tp,
    resume=True,
    replace_output_head=True)

predictor = ttf.Predictor(checkpoint='out/finetune_ckpt.pt')
predictor.predict(data_reader=IncomeDataReader(split['ssl_test']),
                  save_as="prediction_output.csv")
