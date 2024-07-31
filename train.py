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
from tabular_transformer import TabularTransformer, ModelArgs, Task, HyperParameters

# load the default HyperParameters
hp = HyperParameters()
# override the hyperparameters from cli arguments
hp.config_from_cli()

config = hp.asdict()  # will be useful for logging
# -----------------------------------------------------------------------------


# validating checks
assert hp.feature_vocab_size == 2048, "The feature vocab size for TabularTransformer is 2048 now"
assert hp.max_seq_len == hp.num_cols, "max_seq_len must equal num_cols"
assert hp.loss_type in (1, 2, 3, 4)
assert hp.pretext_target_col is None or hp.loss_type == 4, "use contrastive learn pretext target col, loss_type must be 4 (SUPCON LOSS)"

# we are running on a single gpu, and one process
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = hp.gradient_accumulation_steps * \
    ddp_world_size * hp.batch_size * hp.max_seq_len

print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"breaks down as: {hp.gradient_accumulation_steps} grad accum steps * {
      ddp_world_size} processes * {hp.batch_size} batch size * {hp.max_seq_len} max seq len")

os.makedirs(hp.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

# when enabled, pyTorch is allowed to use the TensorFloat32 (TF32) tensor cores
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# for later use in torch.autocast
device_type = "cuda" if "cuda" in hp.device else "cpu"

# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32,
           "bfloat16": torch.bfloat16, "float16": torch.float16}[hp.dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=hp.batch_size,
    max_seq_len=hp.max_seq_len,
    feature_vocab_size=hp.feature_vocab_size,
    datafile=hp.data_file,
    min_cat_count=hp.min_cat_count,
    validate_split=hp.validate_split,
    apply_power_transform=hp.apply_power_transform,
    remove_outlier=hp.remove_outlier,
    device=hp.device,
    pretext_with_label=hp.pretext_with_label,
    pretext_target_col=hp.pretext_target_col,
    pretext_col_unk_ratio=hp.pretext_col_unk_ratio,
    num_workers=0,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=hp.dim,
    n_layers=hp.n_layers,
    n_heads=hp.n_heads,
    loss_type=hp.loss_type,
    feature_vocab_size=hp.feature_vocab_size,
    output_dim=hp.output_dim,
    output_hidden_dim=hp.output_hidden_dim,
    output_forward_dim=hp.output_forward_dim,
    multiple_of=hp.multiple_of,
    max_seq_len=hp.max_seq_len,
    dropout=hp.dropout,
    finetune=hp.finetune,
)  # start with model_args from command line


if hp.init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = TabularTransformer(gptconf)
elif hp.init_from in ("resume", "resume_with_new_head"):
    print(f"Resuming training from {hp.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(
        hp.out_dir, hp.input_checkpoint if hp.input_checkpoint is not None else hp.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=hp.device)
    checkpoint_model_args = checkpoint["model_args"]

    fused_model_args = {}
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "feature_vocab_size", "multiple_of",
              "max_seq_len", "output_forward_dim", "output_hidden_dim"]:
        assert model_args[k] == checkpoint_model_args[k], f"model args: {
            k} not consistent with checkpoint"
        fused_model_args[k] = model_args[k]

    assert not hp.init_from == "resume" or model_args["loss_type"] == checkpoint_model_args["loss_type"]
    fused_model_args["loss_type"] = model_args["loss_type"]

    assert not hp.init_from == "resume" or model_args[
        "output_dim"] == checkpoint_model_args["output_dim"]
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
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # load state dict
    model.load_state_dict(state_dict)

    # reset output head
    if hp.init_from == "resume_with_new_head":
        model.reset_output_head(model_args["output_dim"])

    if not hp.reset_iter_when_resume:
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"bad init_from value: {hp.init_from}")


# load model to device
model.to(hp.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(hp.dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    hp.weight_decay, hp.learning_rate, (hp.beta1, hp.beta2), device_type)
if hp.init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if hp.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

loss_estimate_rng = random.Random(hp.dataset_seed)
train_rng = random.Random(hp.dataset_seed)
# helps estimate an arbitrarily accurate loss over either split using many batches


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        k = 0
        losses = torch.zeros(hp.eval_iters)  # keep on CPU
        while (k < hp.eval_iters):
            seed = loss_estimate_rng.randint(1024, 1024*1024)
            batch_iter = iter_batches(
                split=split, seed=seed, unk_ratio=hp.unk_ratio)
            for X, Y in batch_iter:
                with ctx:
                    logits = model(X, Y)
                    loss = raw_model.last_loss
                losses[k] = loss.item()
                k += 1
                if k >= hp.eval_iters:
                    break
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)


def get_lr(it):
    # finetune
    if hp.finetune:
        return hp.learning_rate
    # 1) linear warmup for warmup_iters steps
    if it < hp.warmup_iters:
        return hp.learning_rate * it / hp.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - hp.warmup_iters) / (lr_decay_iters - hp.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (hp.learning_rate - min_lr)


# logging
if hp.wandb_log and master_process:
    import wandb
    wandb.init(project=hp.wandb_project, name=hp.wandb_run_name, config=config)

# create iter batch only to instantiate the SingletonDataset
_ = iter_batches(split="train", seed=hp.dataset_seed, unk_ratio=None)

num_batches_per_epoch = Task.get_dataset_attributes(
)['train_dataset_size'] // hp.batch_size

# fixing some hyperparams to sensible defaults
# should be ~= max_iters per Chinchilla
lr_decay_iters = hp.max_epochs * num_batches_per_epoch
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
for _epoch in range(hp.max_epochs):
    batch_seed = train_rng.randint(1024, 1024*1024)
    train_batch_iter = iter_batches(
        split="train", seed=batch_seed, unk_ratio=hp.unk_ratio)
    # iterate over the dataset
    for X, Y in train_batch_iter:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if hp.decay_lr else hp.learning_rate
        if not raw_model.finetune:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % hp.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {
                losses['train']:.4f}, val loss {losses['val']:.4f}")
            if hp.wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }, step=iter_num
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or hp.always_save_checkpoint:
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
                    print(f"saving checkpoint to {hp.out_dir}")
                    torch.save(checkpoint, os.path.join(
                        hp.out_dir, hp.output_checkpoint if hp.output_checkpoint is not None else hp.checkpoint))

        if iter_num == 0 and hp.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(hp.gradient_accumulation_steps):
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = loss / hp.gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if hp.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), hp.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % hp.log_interval == 0 and master_process:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * hp.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    hp.batch_size * hp.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {
                    dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

    # two nested loops need to break twice
    if hp.eval_only:
        break
