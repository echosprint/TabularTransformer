from contextlib import nullcontext
from typing import Literal, Optional
import torch
import pandas as pd
from .data_common import DataReader
from .preprocessor import normalize_data, power_transform, preprocess
from .tabular_transformer import ModelArgs, TabularTransformer
from .tokenizer import Tokenizer
from .dataloader import load_data
import random
from .util import LossType, TaskType
import numpy as np
from .metrics import calAUC, calAccuracy, calF1Macro, calMAPE, calRMSE
from pathlib import Path
import torch.nn.functional as F


class Predictor:
    data_reader: DataReader
    checkpoint: str
    seed: int
    device_type: Literal['cuda', 'cpu']
    has_truth: bool
    batch_size: int
    save_as: Optional[str | Path]

    def __init__(self, checkpoint: str = 'out/ckpt.pt'):
        checkpoint_path = Path(checkpoint)
        assert checkpoint_path.exists(), \
            f"checkpoint file: {checkpoint} not exists. Abort."
        self.checkpoint = checkpoint_path

    def predict(self,
                data_reader: DataReader,
                has_truth: bool = True,
                batch_size: int = 1024,
                save_as: Optional[str | Path] = None,
                seed: int = 1337):

        assert isinstance(data_reader, DataReader)
        self.data_reader = data_reader
        self.seed = seed
        self.has_truth = has_truth
        self.batch_size = batch_size
        self.save_as = save_as
        assert str(self.save_as).endswith('.csv'), \
            "only support save as .csv file"

        self._initialize()

        self._load_checkpoint()

        self._init_model()

        self._init_tokenizer_dataset()

        self._predict()

        self._post_process()
        # if self.truth_y is not None:
        #     self._cal_loss(self.logits_array)

        if self.save_as is not None:
            self._save_output()

    def _initialize(self):
        # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_available() \
            and torch.cuda.is_bf16_supported() else 'float32'  # 'float32' or 'bfloat16' or 'float16'

        # for later use in torch.autocast
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]

        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        if self.device_type == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=ptdtype)

        self.p_prob = []
        self.p_pred = []
        self.p_loss = []

    def _load_checkpoint(self):
        # init from a model saved in a specific directory
        checkpoint_dict = torch.load(
            self.checkpoint, map_location=self.device)
        print(f"load checkpoint from {self.checkpoint}")
        self.checkpoint_dict = checkpoint_dict
        self.dataset_attr = checkpoint_dict['features']
        self.train_config = checkpoint_dict['config']
        self.model_args = checkpoint_dict['model_args']

    def _init_model(self):
        self.model = TabularTransformer(self.model_args)

        state_dict = self.checkpoint_dict['model']

        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        self.model.to(self.device)

    def _init_tokenizer_dataset(self):
        # load the tokenizer
        self.enc = Tokenizer(self.dataset_attr['feature_vocab'],
                             self.dataset_attr['feature_type'])

        self.loss_type = LossType[self.model_args.loss_type]
        assert self.loss_type is not LossType.SUPCON, \
            "model trained with `SUPCON` loss cannnot be used to predict"

        self.target_map = self.dataset_attr['target_map']
        self.target_stats = self.dataset_attr['target_stats']
        self.task_type = self.dataset_attr['task_type']

        self.predict_map = {v: k for k, v in self.target_map.items()}

        predict_dataframe = self.data_reader.read_data_file()

        if self.has_truth:
            self.dataset_x = predict_dataframe.iloc[:, :-1]
            self.truth_y = predict_dataframe.iloc[:, -1]

        else:
            self.dataset_x = predict_dataframe
            self.truth_y = None

        assert self.dataset_x.shape[1] == self.dataset_attr['max_seq_len']

    def _predict(self):

        self.logits_array = np.zeros(len(self.dataset_x), dtype=float)

        # run generation
        with torch.no_grad():
            with self.ctx:
                num_batches = (len(self.dataset_x) +
                               self.batch_size - 1) // self.batch_size
                for ix in range(num_batches):
                    # encode the beginning of the prompt
                    start = ix * self.batch_size
                    end = start + self.batch_size

                    x = self.dataset_x[start: end]

                    truth = self.truth_y[start: end] if self.truth_y is not None else None

                    # preprocess the data
                    xp = preprocess(self.rng,
                                    x,
                                    self.dataset_attr['feature_type'],
                                    self.dataset_attr['feature_stats'],
                                    self.train_config['apply_power_transform'],
                                    self.train_config['remove_outlier'],
                                    )
                    tok_x = self.enc.encode(xp)
                    feature_tokens = tok_x[0].to(
                        self.device, non_blocking=True)
                    feature_weight = tok_x[1].to(
                        self.device, non_blocking=True)

                    truth_tok = self._preprocess_target(truth) \
                        if truth is not None else None

                    logits, loss = self.model.predict(
                        (feature_tokens, feature_weight), truth_tok)

                    if loss is not None:
                        self.p_loss.append(loss.item())

                    if self.loss_type is LossType.BINCE:
                        bin_prob = torch.sigmoid(logits).squeeze(-1)
                        bin_predict = (bin_prob >= 0.5).long()

                        self.p_prob.append(bin_prob.to(
                            'cpu', dtype=torch.float32) .numpy())
                        self.p_pred.append(bin_predict.to('cpu').numpy())

                    elif self.loss_type is LossType.MULCE:
                        mul_prob = F.softmax(logits, dim=1)
                        mul_predict = torch.argmax(mul_prob, dim=1)

                        self.p_prob.append(mul_prob.to(
                            'cpu', dtype=torch.float32).numpy())
                        self.p_pred.append(mul_predict.to('cpu').numpy())

                    elif self.loss_type is LossType.MSE:
                        u_logits = logits * self.target_stats.logstd + self.target_stats.logmean
                        reg_predict = torch.where(u_logits > 0,
                                                  torch.expm1(u_logits), -torch.expm1(-u_logits))

                        self.p_pred.append(reg_predict.to(
                            'cpu', dtype=torch.float32).numpy())

    def _post_process(self):

        self.predict_results = np.concatenate(self.p_pred, axis=0)

        self.predict_results_output = pd.DataFrame(
            self.predict_results,
            columns=['prediction_outputs']
        )

        if self.task_type is not TaskType.REGRESSION:
            self.probs = np.concatenate(self.p_prob, axis=0)
            self.predict_results_output['prediction_outputs'] = \
                self.predict_results_output['prediction_outputs'].apply(
                    lambda x: self.predict_map[x])

        else:
            self.probs = None

        if self.has_truth:

            self.losses = np.array(self.p_loss).mean()

            if self.task_type is TaskType.BINCLASS:
                self._process_bin()
            elif self.task_type is TaskType.MULTICLASS:
                self._process_mul()
            elif self.task_type is TaskType.REGRESSION:
                self._process_mse()
            else:
                raise ValueError(f"bad task_type: {self.task_type}")

    def _process_bin(self):

        truth_y = self.truth_y.map(
            lambda x: self.target_map[x]).to_numpy()

        bce_loss = self.losses
        print(f"binary cross entropy loss: {bce_loss:.4f}")

        auc_score = calAUC(truth_y, self.probs)
        print(f"auc score: {auc_score:.4f}")

        f1_score = calF1Macro(truth_y, self.predict_results)
        print(f"f1 macro score: {f1_score:.4f}")

        accuracy = calAccuracy(truth_y, self.predict_results)
        print(f"samples: {len(truth_y)}, "
              f"accuracy: {accuracy:.4f}")

    def _process_mul(self):
        truth_y = self.truth_y.map(
            lambda x: self.target_map[x]).to_numpy()

        ce_loss = self.losses
        print(f"cross entropy loss: {ce_loss:.4f}")

        auc_score = calAUC(truth_y, self.probs)
        print(f"auc score: {auc_score:.4f}")

        f1_score = calF1Macro(truth_y, self.predict_results)
        print(f"f1 macro score: {f1_score:.4f}")

        accuracy = calAccuracy(truth_y, self.predict_results)
        print(f"samples: {len(truth_y)}, "
              f"accuracy: {accuracy:.4f}")

    def _process_mse(self):
        truth_y = self.truth_y.to_numpy()

        log_mse_loss = self.losses
        print(f"mse loss of normalized log1p(y): {log_mse_loss:.4f}")

        mape = calMAPE(truth_y, self.predict_results)
        print(f"mean absolute percentage error: {mape:.4f}")

        rmse = calRMSE(truth_y, self.predict_results)
        print(f"root mean square error: {rmse:.4f}")

    def _preprocess_target(self, truth):
        if self.task_type is TaskType.REGRESSION:
            truth_tok = truth.map(lambda x: normalize_data(
                power_transform(x),
                self.target_stats.logmean,
                self.target_stats.logstd))
            target_dtype = torch.float32
        else:
            truth_tok = truth.map(lambda x: self.target_map[x])
            target_dtype = torch.long
        truth_tok_tensor = torch.tensor(
            truth_tok.to_numpy(), dtype=target_dtype).squeeze()
        return truth_tok_tensor.to(self.device, non_blocking=True)

    def _save_output(self):
        output_dir = Path(self.train_config['out_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / self.save_as
        self.predict_results_output.to_csv(filepath, index=False)
        print(f"save prediction output to file: {filepath}")
