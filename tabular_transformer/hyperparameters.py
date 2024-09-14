from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from tabular_transformer.data_common import DataclassTool


@dataclass
class HyperParameters(DataclassTool):
    """
    Hyperparameters for Transformer model and AdamW optimizer 

    Attributes:
        dim (int): Dimension of embedding. Default is 64.
        n_layers (int): Number of Transformer layers. Default is 6.
        n_heads (int): Number of attention heads. Default is 8.
        output_hidden_dim (int): Hidden layer dimension of output MLP head. Default is 128.
        output_forward_dim (int): Dimension to squeeze the embedding before concatenation. Default is 8.
        multiple_of (int): Hidden dimension will be a multiple of this value. Default is 32.
        dropout (float): Dropout ratio. Default is 0.0.
        weight_decay (float): Weight decay parameter in AdamW optimizer. Default is 0.1.
        beta1 (float): Beta1 parameter in AdamW optimizer. Default is 0.9.
        beta2 (float): Beta2 parameter in AdamW optimizer. Default is 0.95.
        grad_clip (float): Gradient clipping value; disable if set to 0.0. Default is 1.0.
    """
    dim: int = 64
    n_layers: int = 6
    n_heads: int = 8
    output_hidden_dim: int = 128
    output_forward_dim: int = 8
    multiple_of: int = 32
    dropout: float = 0.0
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class TrainSettings(DataclassTool):
    """
    Training settings and configurations.
    Attributes:
        out_dir (str): Output directory for checkpoints and predictions. Default is "out".
        log_interval (int): Interval of iterations for logging to the terminal. Default is 1.
        eval_only (bool): If True, the script exits after the first evaluation. Default is False.
        wandb_log (bool): Enable logging with Weights & Biases. Default is False.
        wandb_project (str): Weights & Biases project name. Default is "TabularTransformer".
        wandb_run_name (str): Weights & Biases run name. Default is "run".
        min_cat_count (float): Minimum category count for valid classes; others labeled as `UNKNOWN`. Default is 0.02.
        apply_power_transform (bool): Apply power transform to numerical columns. Default is True.
        unk_ratio_default (float): Default percentage of tabular values to be randomly masked as unknown during training. Default is 0.2.
        dataset_seed (int): Seed for dataset loader. Default is 42.
        torch_seed (int): Seed for PyTorch. Default is 1377.
        dataset_device (str): Device to load the dataset when tokenized. Default is "cpu".
        device (str): Training device (e.g., 'cpu', 'cuda'). Default is "cuda".
        dtype (Literal): PyTorch data type for training ('float32', 'bfloat16', 'float16'). Default is "bfloat16".
        compile (bool): Compile model using PyTorch 2.0 for speed. Default is False.
    """
    out_dir: str = "out"
    log_interval: int = 1
    eval_only: bool = False
    wandb_log: bool = False
    wandb_project: str = "TabularTransformer"
    wandb_run_name: str = "run"
    min_cat_count: float = 0.02
    apply_power_transform: bool = True
    unk_ratio_default: float = 0.2
    dataset_seed: int = 42
    torch_seed: int = 1377
    dataset_device: str = "cpu"
    device: str = "cuda"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    compile: bool = False


@dataclass
class TrainParameters(DataclassTool):
    """
    Parameters for the training process.

    Attributes:
        max_iters (int): Total number of training iterations. Default is 100000.
        batch_size (int): Batch size per iteration. Default is 128.
        output_dim (int): Output dimension of the model. Default is 1.
        loss_type (Literal): Type of loss function ('BINCE', 'MULCE', 'MSE', 'SUPCON'). 
            `BINCE`: `torch.nn.functional.binary_cross_entropy_with_logits`,
            `MULCE`: `torch.nn.functional.cross_entropy`,
            `MSE`: `torch.nn.functional.mse_loss`,
            `SUPCON`: `Supervised Contrastive Loss`, see arXiv:2004.11362,
            Default is 'BINCE'.
        eval_interval (int): Interval of iterations to start an evaluation. Default is 100.
        eval_iters (int): Number of iterations to run during evaluation. Default is 100.
        validate_split (float): Proportion of training data used for validation. Default is 0.2.
        unk_ratio (Dict[str, float]): Unknown ratio for specific columns, overrides `unk_ratio_default`. Default is `{}`.
        learning_rate (float): Learning rate for the optimizer. Default is 5e-4.
        transformer_lr (float): Learning rate for the transformer part; overrides `learning_rate` if set. Default is `None`.
        output_head_lr (float): Learning rate for the output head; overrides `learning_rate` if set. Default is `None`.
        warmup_iters (int): Number of iterations for learning rate warm-up. Default is 1000.
        lr_scheduler (Literal): Type of learning rate scheduler ('constant', 'cosine'). Default is 'cosine'.
        checkpoint (str): Checkpoint file name for saving and loading. Default is "ckpt.pt".
        input_checkpoint (str): Input checkpoint file for resuming training, overrides `checkpoint` if set.
        output_checkpoint (str): Output checkpoint file name for saving, overrides `checkpoint` if set.
        always_save_checkpoint (bool): Always save checkpoint regardless of evaluation results. Default is False.
    """
    max_iters: int = 100000
    batch_size: int = 128
    output_dim: int = 1
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON'] = 'BINCE'
    eval_interval: int = 100
    eval_iters: int = 100
    validate_split: float = 0.2
    unk_ratio: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 5e-4
    transformer_lr: float = None
    output_head_lr: float = None
    warmup_iters: int = 1000
    lr_scheduler: Literal['constant', 'cosine'] = 'cosine'
    checkpoint: str = "ckpt.pt"
    input_checkpoint: str = None
    output_checkpoint: str = None
    always_save_checkpoint: bool = False


@dataclass
class ModelArgs(DataclassTool):
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 8
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON'] = 'BINCE'  # noqa: E501
    feature_vocab_size: int = 2048
    output_dim: int = 1
    output_hidden_dim: int = 128
    output_forward_dim: int = 8
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0
