from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from tabular_transformer.data_common import DataclassTool


@dataclass
class HyperParameters(DataclassTool):
    # dimension of embedding
    dim: int = 64
    # number layers of Transformer blocks
    n_layers: int = 6
    # number of attention heads
    n_heads: int = 8
    # hidden layer dimension of output MLP head
    output_hidden_dim: int = 128
    # squeeze the embedding dim to small `output_forward_dim` before concatenate all features as input for MLP
    output_forward_dim: int = 8
    # make the hidden dim be multiple of
    multiple_of: int = 32
    # dropout ratio
    dropout: float = 0.0
    # weight decay in AdamW
    weight_decay: float = 1e-1
    # beta1 in AdamW
    beta1: float = 0.9
    # beta2 in AdamW
    beta2: float = 0.95
    # clip gradients at this value, or disable if == 0.0
    grad_clip: float = 1.0


@dataclass
class TrainSettings(DataclassTool):
    # output dir for checkpoints, predictions
    out_dir: str = "out"
    # interval of iters for log print in terminal
    log_interval: int = 1
    # if True, script exits right after the first eval
    eval_only: bool = False
    # wandb logging
    wandb_log: bool = False
    # wandb project name
    wandb_project: str = "TabularTransformer"
    # wandb run name
    wandb_run_name: str = "run"
    # for categorical columns, the frequency of a class larger than `min_cat_count` will be consider a valid class, otherwise labeled as `UNKNOWN`
    min_cat_count: float = 0.02
    # apply power transform for numerical columns
    apply_power_transform: bool = True
    # default unk ratio for training if not set in `unk_ratio` dict
    unk_ratio_default: float = 0.2
    # seed for dataset loader
    dataset_seed: int = 42
    # seed for torch
    torch_seed: int = 1377
    # load dataset on `dataset_device` when tokenized
    dataset_device: str = "cpu"
    # train device, e.g. `cpu`, `cuda`, `cuda:0`, `cuda:1` etc., or try `mps` on macbooks
    device: str = "cuda"
    # pytorch dtype: `float32` | `bfloat16` | `float16`
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    # use PyTorch 2.0 to compile the model to be faster, comiple not work on Python 3.12+
    compile: bool = False


@dataclass
class TrainParameters(DataclassTool):
    # total number of training iterations
    max_iters: int = 100000
    # batch size per iter
    batch_size: int = 128
    # output dimension
    output_dim: int = 1
    # train loss function: `binary cross entropy`, `cross entropy`, `mean squared error`, `supervised contrastive loss`
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON'] = 'BINCE'  # noqa: E501
    # interval of iters to start an evaluation
    eval_interval: int = 100
    # iters run for evaluate the model
    eval_iters: int = 100
    # split ratio of train data for validation
    validate_split: float = 0.2
    # specify the unknown ratio of col, override the `unk_ratio_default`
    unk_ratio: Dict[str, float] = field(default_factory=dict)
    # learning rate
    learning_rate: float = 5e-4
    # transformer part learning rate, if set, override the `learning_rate`
    transformer_lr: float = None
    # output head part learning rate, if set, override the `learning_rate`
    output_head_lr: float = None
    # how many steps to warm up for
    warmup_iters: int = 1000
    # learning rate scheduler
    lr_scheduler: Literal['constant', 'cosine'] = 'cosine'
    # default checkpoint file name
    checkpoint: str = "ckpt.pt"
    # input checkpoint for resume training, if set, override `checkpoint`
    input_checkpoint: str = None
    # output checkpoint for checkpoint save, if set, override `checkpoint`
    output_checkpoint: str = None
    # always save checkpoint no matter the evaluation is good or bad
    always_save_checkpoint: bool = False


@dataclass
class ModelArgs(DataclassTool):
    # dimension of embedding
    dim: int = 1024
    # number layers
    n_layers: int = 16
    # attention head
    n_heads: int = 8
    # loss function
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON'] = 'BINCE'  # noqa: E501
    # sum of cardinality of categorical feature and numerical feature, plus [`UNKNOWN`] for each feature
    feature_vocab_size: int = 2048
    # final out dimension
    output_dim: int = 1
    # hidden dim in output head
    output_hidden_dim: int = 128
    # squeeze the embedding dim to `output_forward_dim` before concatenate
    output_forward_dim: int = 8
    # hidden_dim in MLP
    hidden_dim: Optional[int] = None
    # MLP hidden layer size will be multiple of
    multiple_of: int = 256
    # eps added when normalization
    norm_eps: float = 1e-5
    # max columns of tabular data
    max_seq_len: int = 1024
    # drop out
    dropout: float = 0.0
