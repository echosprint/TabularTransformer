"""
This training script can be run on a single gpu in debug mode,

To run on a single GPU small debug run, example:
$ python train.py --compile=False --eval_iters=10 --batch_size=8
"""

import math
import os
import time
import random
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from tabular_transformer import TabularTransformer, ModelArgs
from dataloader import Task

"""
    TODO:
    Pretrain: at least two round pretrain steps, do not remove column when use a column as pretext label,
              just mask it as UNK
    Finetune: keep all the pretrain pretext column label
"""

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'resume_with_new_head'
checkpoint = "ckpt.pt"
reset_iter_when_resume = True # reset the iter number when resume

# wandb logging
wandb_log = True  # disabled by default
wandb_project = "data_analyst_skill_competition"
wandb_run_name = "run_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# data
data_file = "income/income_evaluation_train"
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
feature_vocab_size = 2048 # the dataset feature vocab size must <= transformer feature vocab size
num_cols = 14
max_seq_len = num_cols
min_cat_count = 0.02
apply_power_transform=True
remove_outlier=False
unk_ratio = 0.2 
dataset_seed = 42
loss_type = 1 # BINCLASS = 1 MULTICLASS = 2 REGRESSION MSE = 3 SUPCON = 4
validate_split = 0.2

# pretrain (predict train / contrastive train) 
pretext_with_label = True    # whether pretrain dataset has label in last column
pretext_target_col = None   # pretrain pretext target column name
pretext_col_unk_ratio = 0.75


# model
dim = 1024
n_layers = 16
n_heads = 8
output_dim = 1 # final out dimension
output_hidden_dim = 128
output_forward_dim = 8
multiple_of = 32
dropout = 0.0
finetune = False

# adamw optimizer
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for

# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster, comiple not work on Python 3.12+

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert feature_vocab_size == 2048, "The feature vocab size for TabularTransformer is 2048 now"
assert max_seq_len == num_cols, "max_seq_len must equal num_cols"
assert loss_type in (1, 2, 3, 4)
assert pretext_target_col is None or loss_type == 4, "use contrastive learn pretext target col, loss_type must be 4 (SUPCON LOSS)"

# we are running on a single gpu, and one process
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len

print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

# when enabled, pyTorch is allowed to use the TensorFloat32 (TF32) tensor cores
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    feature_vocab_size=feature_vocab_size,
    datafile=data_file,
    min_cat_count=min_cat_count,
    validate_split=validate_split,
    apply_power_transform=apply_power_transform,
    remove_outlier=remove_outlier,
    device=device,
    pretext_with_label=pretext_with_label,
    pretext_target_col=pretext_target_col,
    pretext_col_unk_ratio=pretext_col_unk_ratio,
    num_workers=0,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    loss_type=loss_type,
    feature_vocab_size=feature_vocab_size,
    output_dim=output_dim,
    output_hidden_dim=output_hidden_dim,
    output_forward_dim=output_forward_dim,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    finetune=finetune,
)  # start with model_args from command line



if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = TabularTransformer(gptconf)
elif init_from in ("resume", "resume_with_new_head"):
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    fused_model_args = {}
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "feature_vocab_size", "multiple_of",
              "max_seq_len", "output_forward_dim", "output_hidden_dim"]:
        assert model_args[k] == checkpoint_model_args[k], f"model args: {k} not consistent with checkpoint"
        fused_model_args[k] = model_args[k]

    assert not init_from == "resume" or model_args["loss_type"] == checkpoint_model_args["loss_type"]
    fused_model_args["loss_type"] = model_args["loss_type"]

    assert not init_from == "resume" or model_args["output_dim"] == checkpoint_model_args["output_dim"]
    fused_model_args["output_dim"] = checkpoint_model_args["output_dim"]

    for param in ['dropout', 'finetune']:
        fused_model_args[param] = model_args[param]

    # create the model
    gptconf = ModelArgs(**fused_model_args)
    model = TabularTransformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    # load state dict
    model.load_state_dict(state_dict)

    # reset output head
    if init_from == "resume_with_new_head":
        model.reset_output_head(model_args["output_dim"])

    if not reset_iter_when_resume:
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"bad init_from value: {init_from}")


# load model to device
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

loss_estimate_rng = random.Random(19324325383)
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        seed = loss_estimate_rng.randint(1024, 1024*1024)
        batch_iter = iter_batches(split=split, seed=seed, unk_ratio=unk_ratio) # make seed to estimate loss on different data
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # finetune
    if finetune:
        return learning_rate
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_iter = iter_batches(split="train", seed=dataset_seed, unk_ratio=unk_ratio)
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    if not raw_model.finetune:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "dataset_attr": Task.get_dataset_attributes(),
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
