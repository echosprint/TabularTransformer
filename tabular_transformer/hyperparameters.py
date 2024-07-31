from dataclasses import dataclass, fields, asdict
from typing import Literal, get_type_hints
from datetime import datetime
from .util import DataclassTool


@dataclass
class HyperParameters(DataclassTool):
    # I/O
    out_dir: str = "out"
    eval_interval: int = 1000
    log_interval: int = 1
    eval_iters: int = 100
    eval_only: bool = False  # if True, script exits right after the first eval
    # if True, always save a checkpoint after each eval
    always_save_checkpoint: bool = False
    init_from: Literal["scratch", "resume", "resume_with_new_head"] = "scratch"
    checkpoint: str = "ckpt.pt"
    input_checkpoint: str = None
    output_checkpoint: str = None
    reset_iter_when_resume: bool = True  # reset the iter number when resume

    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "TabularTransformer"
    wandb_run_name: str = "run_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # data
    data_file: str = "income/income_evaluation_train"
    batch_size: int = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
    # the dataset feature vocab size must <= transformer feature vocab size
    feature_vocab_size: int = 2048
    num_cols: int = 14
    max_seq_len: int = 14
    min_cat_count: float = 0.02
    apply_power_transform: bool = True
    remove_outlier: bool = False
    unk_ratio: float = 0.2
    dataset_seed: int = 42

    # train
    # BINCLASS = 1 MULTICLASS = 2 REGRESSION MSE = 3 SUPCON = 4
    loss_type: Literal[1, 2, 3, 4] = 1
    validate_split: float = 0.2
    # pretrain (predict train / contrastive train)
    pretext_with_label: bool = True  # whether pretrain dataset has label in last column
    pretext_target_col: str = None  # pretrain pretext target column name
    pretext_col_unk_ratio: float = 0.75

    # model
    dim: int = 64
    n_layers: int = 6
    n_heads: int = 8
    output_dim: int = 1  # final out dimension
    output_hidden_dim: int = 128
    output_forward_dim: int = 8
    multiple_of: int = 32
    dropout: float = 0.0
    finetune: bool = False

    # adamw optimizer
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    learning_rate: float = 5e-4  # max learning rate
    max_epochs: int = 200  # total number of training epochs
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for

    # system
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device: str = "cuda"
    # float32|bfloat16|float16
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    # use PyTorch 2.0 to compile the model to be faster, comiple not work on Python 3.12+
    compile: bool = False
