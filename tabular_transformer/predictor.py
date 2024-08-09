from contextlib import nullcontext
from typing import Literal, Optional
import torch
import pandas as pd
from .data_common import DataReader
from .preprocessor import preprocess
from .tabular_transformer import ModelArgs, TabularTransformer
from .tokenizer import Tokenizer
from .dataloader import load_data
import random
from .util import LossType
import numpy as np
from .metrics import calAUC
from pathlib import Path


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
                batch_size: int = 128,
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

        if self.truth_y is not None:
            self._cal_loss(self.logits_array)

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

    def _load_checkpoint(self):
        # init from a model saved in a specific directory
        checkpoint_dict = torch.load(
            self.checkpoint, map_location=self.device)
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

        loss_type = self.model_args.loss_type
        assert LossType[loss_type] is LossType.BINCE, \
            "only support binary cross entropy loss"

        self.target_map = self.dataset_attr['target_map']
        assert self.target_map is not None

        self.predict_map = {v: k for k, v in self.target_map.items()}

        predict_dataframe = self.data_reader.read_data_file()

        if self.has_truth:
            self.dataset_x = predict_dataframe.iloc[:, :-1]
            self.truth_y = predict_dataframe.iloc[:, -1]

        else:
            self.dataset_x = predict_dataframe
            self.truth_y = None
        assert self.dataset_x.shape[1] == self.dataset_attr['max_seq_len']

        self.accuracy_accum = {"n_samples": 0, "right": 0}

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
                    logits = self.model.predict(
                        (feature_tokens, feature_weight))
                    logits_y = logits.squeeze(-1) \
                        .to('cpu', dtype=torch.float32).numpy()

                    self.logits_array[start: end] = logits_y

                    self.accum_accuracy(logits_y, truth.to_numpy()
                                        if truth is not None else None)
        self.predict_result_array = self.get_results(self.logits_array)

    def _save_output(self):
        df = pd.DataFrame(self.predict_result_array,
                          columns=['prediction_outputs'])
        output_dir = Path(self.train_config['out_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / self.save_as
        df.to_csv(filepath, index=False)
        print(f"save prediction output to file: {filepath}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_results(self, logits_arr):
        prob_y = self.sigmoid(logits_arr)
        result_val = np.where(prob_y > 0.5, 1, 0)
        result_cls = np.vectorize(self.predict_map.get)(result_val)
        return result_cls

    def accum_accuracy(self, logits, truth=None):
        if truth is None:
            return
        assert len(logits) == len(truth)
        result_cls = self.get_results(logits)
        equal_elements = np.sum(np.equal(result_cls, truth))
        self.accuracy_accum['n_samples'] += len(logits)
        self.accuracy_accum['right'] += equal_elements

    def binary_cross_entropy_loss(self, logits, targets=None):
        # Apply sigmoid to logits
        probs = self.sigmoid(logits)
        targets = np.vectorize(self.target_map.get)(targets)
        # Compute binary cross-entropy loss
        loss = - (targets * np.log(probs) +
                  (1 - targets) * np.log(1 - probs))
        # Return the mean loss
        return np.mean(loss)

    def _cal_loss(self, logits_array):
        bce_loss = self.binary_cross_entropy_loss(
            logits_array, self.truth_y.to_numpy())
        print(f"binary cross entropy loss: {bce_loss:.4f}")
        auc_score = calAUC(self.truth_y.to_numpy(), logits_array)
        print(f"auc score: {auc_score}")
        print(f"samples: {self.accuracy_accum['n_samples']}, "
              f"accuracy: {self.accuracy_accum['right'] / self.accuracy_accum['n_samples']:.2f}")
