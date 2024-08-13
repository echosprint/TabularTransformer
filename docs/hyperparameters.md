## HyperParameters

- **dim** (*int*): dimension of embedding. Default: `64`.
- **n_layers** (*int*): number layers of Transformer blocks. Default: `6`.
- **n_heads** (*int*): number of attention heads. Default: `8`.
- **output_hidden_dim** (*int*): hidden layer dimension of output MLP head. Default: `128`.
- **output_forward_dim** (*int*): squeeze the embedding dim to small output_forward_dim before concatenate all features for ouput MLP head. Default: `8`.
- **multiple_of** (*int*): make the hidden dim be multiple of. Default: `32`.
- **dropout** (*float*): dropout ratio. Default: `0.0`.
- **gradient_accumulation_steps** (*int*): gradient accumulation steps before update, used to simulate larger batch sizes. Default: `1`.
- **weight_decay** (*float*): weight decay in AdamW. Default: `1e-1`.
- **beta1** (*float*): beta1 in AdamW. Default: `0.9`.
- **beta2** (*float*): beta2 in AdamW. Default: `0.95`.
- **grad_clip** (*float*): clip gradients at this value, or disable if == 0.0. Default: `1.0`.


## TrainSettings

- **out_dir** (*str*): output dir for checkpoints, predictions. Default: `"out"`.
- **log_interval** (*int*): interval of iters for log print in terminal. Default: `1`.
- **eval_iters** (*int*): interval of iters for evaluate the model. Default: `100`.
- **eval_only** (*bool*): if True, script exits right after the first eval. Default: `False`.
- **always_save_checkpoint** (*bool*): always save checkpoint no matter the evaluation is good or bad. Default: `False`.
- **wandb_log** (*bool*): wandb logging. Default: `False`.
- **wandb_project** (*str*): wandb project name. Default: `"TabularTransformer"`.
- **wandb_run_name** (*str*): wandb run name. Default: `"run"`.
- **min_cat_count** (*float*): for categorical columns, the frequency of a class larger than `min_cat_count` will be consider a valid class, otherwise labeled as `UNKNOWN`. Default: `0.02`.
- **apply_power_transform** (*bool*): apply power transform for numerical columns. Default: `True`.
- **remove_outlier** (*bool*): remove outliers. Default: `False`.
- **unk_ratio_default** (*float*): default unk ratio for training if not set in `unk_ratio` dict. Default: `0.2`.
- **dataset_seed** (*int*): seed for dataset loader. Default: `42`.
- **device** (*str*): train device, e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks. Default: `"cuda"`.
- **dtype** (*Literal["float32", "bfloat16", "float16"]*): pytorch dtype: float32|bfloat16|float16. Default: `"bfloat16"`.
- **compile** (*bool*): use PyTorch 2.0 to compile the model to be faster, comiple not work on Python 3.12+. Default: `False`.


## TrainParameters

- **train_epochs** (*int*): train epochs for a dataset. Default: `200`.
- **batch_size** (*int*): batch size per iter. Default: `128`.
- **output_dim** (*int*): output dimension. Default: `1`.
- **loss_type** (*Literal['BINCE', 'MULCE', 'MSE', 'SUPCON']*): train loss function: binary cross entropy, cross entropy, mean squared error, supervised contrastive loss. Default: `'BINCE'`.
- **eval_interval** (*int*): interval of iters to start an evaluation. Default: `100`.
- **validate_split** (*float*): split ratio of train data for validation. Default: `0.2`.
- **unk_ratio** (*Dict[str, float]*): specify the unknown ratio of col, override the unk_ratio_default. Default: `field(default_factory=dict)`.
- **learning_rate** (*float*): learning rate. Default: `5e-4`.
- **transformer_lr** (*float*): transformer part learning rate, if set, override the `learning_rate`. Default: `None`.
- **output_head_lr** (*float*): output head part learning rate, if set, override the `learning_rate`. Default: `None`.
- **warmup_iters** (*int*): how many steps to warm up for. Default: `1000`.
- **lr_scheduler** (*Literal['constant', 'cosine']*): learning rate scheduler. Default: `'cosine'`.
- **checkpoint** (*str*): default checkpoint file name. Default: `"ckpt.pt"`.
- **input_checkpoint** (*str*): input checkpoint for resume training, if set, override `checkpoint`. Default: `None`.
- **output_checkpoint** (*str*): output checkpoint for checkpoint save, if set, override `checkpoint`. Default: `None`.


