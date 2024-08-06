from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Literal, Optional, get_type_hints
from datetime import datetime
from .data_common import DataclassTool


@dataclass
class HyperParameters(DataclassTool):
    # I/O
    # out_dir: str = "out"
    # eval_interval: int = 1000
    # log_interval: int = 1
    # eval_iters: int = 100
    # eval_only: bool = False  # if True, script exits right after the first eval
    # if True, always save a checkpoint after each eval
    # always_save_checkpoint: bool = False
    # init_from: Literal["scratch", "resume", "resume_with_new_head"] = "scratch"
    # checkpoint: str = "ckpt.pt"
    # input_checkpoint: str = None
    # output_checkpoint: str = None
    # reset_iter_when_resume: bool = True  # reset the iter number when resume

    # wandb logging
    # wandb_log: bool = False  # disabled by default
    # wandb_project: str = "TabularTransformer"
    # wandb_run_name: str = "run_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # data
    # data_file: str = "income/income_evaluation_train"
    # batch_size: int = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
    # the dataset feature vocab size must <= transformer feature vocab size
    # feature_vocab_size: int = 2048
    # num_cols: int = 14
    # # max_seq_len: int = 14
    # min_cat_count: float = 0.02
    # apply_power_transform: bool = True
    # remove_outlier: bool = False
    # unk_ratio: float = 0.2
    # dataset_seed: int = 42

    # train
    # loss_type: Literal['BINCLASS', 'MULTICLASS', 'REGRESSION', 'SUPCON'] = 'BINCLASS'  # noqa: E501
    # validate_split: float = 0.2
    # pretrain (predict train / contrastive train)
    # pretext_with_label: bool = True  # whether pretrain dataset has label in last column
    # pretext_target_col: str = None  # pretrain pretext target column name
    # pretext_col_unk_ratio: float = 0.75

    # model
    dim: int = 64
    n_layers: int = 6
    n_heads: int = 8
    # output_dim: int = 1  # final out dimension
    output_hidden_dim: int = 128
    output_forward_dim: int = 8
    multiple_of: int = 32
    dropout: float = 0.0
    # finetune: bool = False

    # adamw optimizer
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    # learning_rate: float = 5e-4  # max learning rate
    # for pretrained model paras scale lr to very low, 0. means frozen
    # finetune_lr_scaler: float = 0.1
    # max_epochs: int = 200  # total number of training epochs
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # system
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # device: str = "cuda"
    # # float32|bfloat16|float16
    # dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    # # use PyTorch 2.0 to compile the model to be faster, comiple not work on Python 3.12+
    # compile: bool = False


@dataclass
class TrainSettings(DataclassTool):
    # I/O
    out_dir: str = "out"
    log_interval: int = 1

    eval_iters: int = 100
    eval_only: bool = False  # if True, script exits right after the first eval

    always_save_checkpoint: bool = False

    reset_iter_when_resume: bool = True  # reset the iter number when resume

    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "TabularTransformer"
    # wandb_run_name: str = "run_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb_run_name: str = "run"

    # num_cols: int = 14
    # max_seq_len: int = 14
    min_cat_count: float = 0.02
    apply_power_transform: bool = True
    remove_outlier: bool = False
    unk_ratio_default: float = 0.2
    dataset_seed: int = 42

    # system
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device: str = "cuda"
    # float32|bfloat16|float16
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    # use PyTorch 2.0 to compile the model to be faster, comiple not work on Python 3.12+
    compile: bool = False


@dataclass
class TrainParameters(DataclassTool):

    train_epochs: int = 200
    batch_size: int = 128
    output_dim: int = 1
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON'] = 'BINCE'  # noqa: E501
    eval_interval: int = 100
    validate_split: float = 0.2
    # pretext_target_col: str = None
    # pretext_with_label: bool = True
    unk_ratio: Dict[str, float] = field(default_factory=dict)

    # learning rate and decay settings
    learning_rate: float = 5e-4
    transformer_lr: float = None
    output_head_lr: float = None
    # decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for
    lr_scheduler: Literal['constant', 'cosine'] = 'cosine'

    # checkpoint
    checkpoint: str = "ckpt.pt"
    input_checkpoint: str = None
    output_checkpoint: str = None


@dataclass
class ModelArgs(DataclassTool):
    # default hyperparameters for the Llama 7B model
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 8
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON'] = 'BINCE'  # noqa: E501
    # sum of cardinality of categorical feature and numerical feature, plus [UNKNOWN] for each feature
    feature_vocab_size: int = 2048
    output_dim: int = 1  # final out dimension
    output_hidden_dim: int = 128
    output_forward_dim: int = 8
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 1024  # max columns of tabular data
    dropout: float = 0.0
    # finetune: bool = False  # enable finetune to adjust the learn rate
