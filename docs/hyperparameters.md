## HyperParameters

Hyperparameters for Transformer model and AdamW optimizer

- **dim** (*int*): Dimension of embedding. Default is 64.
- **n_layers** (*int*): Number of Transformer layers. Default is 6.
- **n_heads** (*int*): Number of attention heads. Default is 8.
- **output_hidden_dim** (*int*): Hidden layer dimension of output MLP head. Default is 128.
- **output_forward_dim** (*int*): Dimension to squeeze the embedding before concatenation. Default is 8.
- **multiple_of** (*int*): Hidden dimension will be a multiple of this value. Default is 32.
- **dropout** (*float*): Dropout ratio. Default is 0.0.
- **weight_decay** (*float*): Weight decay parameter in AdamW optimizer. Default is 0.1.
- **beta1** (*float*): Beta1 parameter in AdamW optimizer. Default is 0.9.
- **beta2** (*float*): Beta2 parameter in AdamW optimizer. Default is 0.95.

## TrainSettings

Training settings and configurations.

- **out_dir** (*str*): Output directory for checkpoints and predictions. Default is "out".
- **log_interval** (*int*): Interval of iterations for logging to the terminal. Default is 1.
- **eval_only** (*bool*): If True, the script exits after the first evaluation. Default is False.
- **wandb_log** (*bool*): Enable logging with Weights & Biases. Default is False.
- **wandb_project** (*str*): Weights & Biases project name. Default is "TabularTransformer".
- **wandb_run_name** (*str*): Weights & Biases run name. Default is "run".
- **min_cat_count** (*float*): Minimum category count for valid classes; others labeled as `UNKNOWN`. Default is 0.02.
- **apply_power_transform** (*bool*): Apply power transform to numerical columns. Default is True.
- **unk_ratio_default** (*float*): Default percentage of tabular values to be randomly masked as unknown during training. Default is 0.2.
- **dataset_seed** (*int*): Seed for dataset loader. Default is 42.
- **torch_seed** (*int*): Seed for PyTorch. Default is 1377.
- **dataset_device** (*str*): Device to load the dataset when tokenized. Default is "cpu".
- **device** (*str*): Training device (e.g., 'cpu', 'cuda'). Default is "cuda".
- **dtype** (*Literal*): PyTorch data type for training ('float32', 'bfloat16', 'float16'). Default is "bfloat16".

## TrainParameters

Parameters for the training process.

- **max_iters** (*int*): Total number of training iterations. Default is 100000.
- **batch_size** (*int*): Batch size per iteration. Default is 128.
- **output_dim** (*int*): Output dimension of the model. Default is 1.
- **loss_type** (*Literal*): Type of loss function ('BINCE', 'MULCE', 'MSE', 'SUPCON').              `BINCE`: `torch.nn.functional.binary_cross_entropy_with_logits`,             `MULCE`: `torch.nn.functional.cross_entropy`,             `MSE`: `torch.nn.functional.mse_loss`,             `SUPCON`: `Supervised Contrastive Loss`, see arXiv:2004.11362,             Default is 'BINCE'.
- **eval_interval** (*int*): Interval of iterations to start an evaluation. Default is 100.
- **eval_iters** (*int*): Number of iterations to run during evaluation. Default is 100.
- **validate_split** (*float*): Proportion of training data used for validation. Default is 0.2.
- **unk_ratio** (*Dict[str, float]*): Unknown ratio for specific columns, overrides `unk_ratio_default`. Default is `{}`.
- **learning_rate** (*float*): Learning rate for the optimizer. Default is 5e-4.
- **transformer_lr** (*float*): Learning rate for the transformer part; overrides `learning_rate` if set. Default is `None`.
- **output_head_lr** (*float*): Learning rate for the output head; overrides `learning_rate` if set. Default is `None`.
- **warmup_iters** (*int*): Number of iterations for learning rate warm-up. Default is 1000.
- **lr_scheduler** (*Literal*): Type of learning rate scheduler ('constant', 'cosine'). Default is 'cosine'.
- **checkpoint** (*str*): Checkpoint file name for saving and loading. Default is "ckpt.pt".
- **input_checkpoint** (*str*): Input checkpoint file for resuming training, overrides `checkpoint` if set.
- **output_checkpoint** (*str*): Output checkpoint file name for saving, overrides `checkpoint` if set.


