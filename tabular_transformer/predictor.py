from contextlib import nullcontext
from typing import Literal, Optional
import torch
import pandas as pd
from tabular_transformer.datareader import DataReader
from tabular_transformer.featurestats import FeatureStats
from tabular_transformer.tabular_dataset import TabularDataset
from tabular_transformer.tabular_transformer import ModelArgs, TabularTransformer
import random
from tabular_transformer.util import LossType, TaskType
import numpy as np
from tabular_transformer.metrics import calAUC, calAccuracy, calF1Macro, calMAPE, calRMSE
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm


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
                batch_size: int = 1024,
                save_as: Optional[str | Path] = None,
                seed: int = 1337):

        assert isinstance(data_reader, DataReader)
        self.data_reader = data_reader
        self.seed = seed
        self.has_truth = True if self.data_reader.label is not None else False
        self.batch_size = batch_size
        self.save_as = save_as
        assert str(self.save_as).endswith('.csv'), \
            "only support save as .csv file"

        self._initialize()

        self._load_checkpoint()

        self._init_model()

        self._init_dataset()

        self._predict()

        self._post_process()

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
            self.checkpoint,
            map_location=self.device,
            weights_only=False,
        )
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

    def _init_dataset(self):

        self.loss_type = LossType[self.model_args.loss_type]
        assert self.loss_type is not LossType.SUPCON, \
            "model trained with `SUPCON` loss cannnot be used to predict"

        self.feature_stats: FeatureStats = self.dataset_attr['feature_stats']
        self.task_type: TaskType = self.dataset_attr['task_type']

        self.predict_map = self.feature_stats.label_cls_map
        self.apply_power_transform = self.train_config['apply_power_transform']

        self.dataset = TabularDataset(
            datareader=self.data_reader,
            device='cpu',
            original_feature_stats=self.feature_stats,
            min_cat_count=1,
            apply_power_transform=self.apply_power_transform,
            validate_split=0,
            seed=42,
        )

        self.dataset_x_tok = self.dataset.dataset_x_tok
        self.dataset_x_val = self.dataset.dataset_x_val
        self.truth_y = self.dataset.dataset_y
        self.id = self.dataset.id
        self.id_name = self.data_reader.id

    def _predict(self):

        self.logits_array = np.zeros(self.dataset_x_tok.size(0), dtype=float)

        # run generation
        with torch.no_grad():
            with self.ctx:
                num_batches = (self.dataset_x_tok.size(0) +
                               self.batch_size - 1) // self.batch_size
                for ix in tqdm(range(num_batches)):
                    # encode the beginning of the prompt
                    start = ix * self.batch_size
                    end = start + self.batch_size

                    x_tok = self.dataset_x_tok[start: end]
                    x_val = self.dataset_x_val[start: end]

                    feature_tokens = x_tok.long().to(
                        self.device, non_blocking=True)
                    feature_weight = x_val.to(
                        self.device, non_blocking=True)

                    if self.truth_y is not None:
                        truth = self.truth_y[start: end]
                        truth = truth if torch.is_floating_point(
                            truth) else truth.long()
                        truth = truth.to(self.device, non_blocking=True)
                    else:
                        truth = None

                    logits, loss = self.model.predict(
                        (feature_tokens, feature_weight), truth)

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
                        itrans_logits = self._inverse_transform(
                            logits.squeeze(-1))
                        self.p_pred.append(itrans_logits.to(
                            'cpu', dtype=torch.float32).numpy())

    def _post_process(self):

        self.predict_results = np.concatenate(self.p_pred, axis=0)

        self.probs = np.concatenate(self.p_prob, axis=0) \
            if len(self.p_prob) > 0 else None
        id_seq = {} if self.id is None else {self.id_name: self.id}
        result = {'prediction': self.predict_results}
        probs_dict = {}
        if self.probs is not None:
            if len(self.probs.shape) == 1:
                probs_dict = {'probability': self.probs}
            else:
                probs_dict = {f'prob_{i}': self.probs[:, i]
                              for i in range(self.probs.shape[1])}

        self.predict_results_output = pd.DataFrame(
            id_seq | result | probs_dict)

        if self.task_type is not TaskType.REGRESSION:
            self.predict_results_output['prediction'] = \
                self.predict_results_output['prediction'].apply(
                    lambda x: self.predict_map[x])

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

        truth_y = self.truth_y.cpu().numpy()

        bce_loss = self.losses
        print(f"binary cross entropy loss: {bce_loss:.6f}")

        auc_score = calAUC(truth_y, self.probs)
        print(f"auc score: {auc_score:.6f}")

        f1_score = calF1Macro(truth_y, self.predict_results)
        print(f"f1 macro score: {f1_score:.6f}")

        accuracy = calAccuracy(truth_y, self.predict_results)
        print(f"samples: {len(truth_y)}, "
              f"accuracy: {accuracy:.4f}")

    def _process_mul(self):
        truth_y = self.truth_y.cpu().numpy()

        ce_loss = self.losses
        print(f"cross entropy loss: {ce_loss:.6f}")

        try:
            auc_score = calAUC(truth_y, self.probs, multi_class=True)
            print(f"auc score: {auc_score:.6f}")
        except ValueError as e:
            print(f"skip cal AUC score due to error: {e}")

        f1_score = calF1Macro(truth_y, self.predict_results)
        print(f"f1 macro score: {f1_score:.6f}")

        accuracy = calAccuracy(truth_y, self.predict_results)
        print(f"samples: {len(truth_y)}, "
              f"accuracy: {accuracy:.4f}")

    def _inverse_transform(self, logits):
        mean, std, mean_log, std_log = self.feature_stats.y_num_stats
        if self.apply_power_transform:
            u_logits = logits.float() * (std_log + 1e-8) + mean_log
            rev_logits = torch.where(u_logits > 0,
                                     torch.expm1(u_logits), -torch.expm1(-u_logits))
        else:
            rev_logits = logits.float() * (std + 1e-8) + mean
        return rev_logits

    def _process_mse(self):
        truth_y = self._inverse_transform(self.truth_y).cpu().numpy()

        log_mse_loss = self.losses
        mse_loss_str = "MSE loss of normalized log1p(y)" \
            if self.apply_power_transform else "mse loss of normalized y"
        print(f"{mse_loss_str}: {log_mse_loss:.6f}")

        mape = calMAPE(truth_y, self.predict_results)
        print(f"mean absolute percentage error: {mape:.4f}")

        rmse = calRMSE(truth_y, self.predict_results)
        print(f"root mean square error: {rmse:.4f}")

    def _save_output(self):
        output_dir = Path(self.train_config['out_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / self.save_as
        self.predict_results_output.to_csv(filepath, index=False)
        print(f"save prediction output to file: {filepath}")
