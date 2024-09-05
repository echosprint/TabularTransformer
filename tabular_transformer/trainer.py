from contextlib import nullcontext
from datetime import datetime
from functools import partial
import math
import os
import random
import time
from typing import Any, Dict, List, Literal, Optional, Union
from tabular_transformer.featurestats import FeatureStats
from tabular_transformer.tabular_dataset import TabularDataset
from tabular_transformer.tabular_loader import Task
from tabular_transformer.util import LossType, TaskType, equals_except
from tabular_transformer.tabular_transformer import TabularTransformer
from tabular_transformer.hyperparameters import HyperParameters, TrainParameters, TrainSettings, ModelArgs
from tabular_transformer.datareader import DataReader
import torch
import inspect
import copy


class Trainer:

    # parameters
    hp: HyperParameters  # hyper parameters
    tp: TrainParameters  # Train parameters
    ts: TrainSettings  # Train Settings
    model_args: ModelArgs

    # dataset
    data_reader: DataReader  # data reader
    dataset: TabularDataset  # dataset

    # model
    model: Optional[TabularTransformer]  # model

    # optimizer
    optimizer: torch.optim.AdamW

    # checkpoint
    output_checkpoint: str  # checkpoint
    input_checkpoint: str
    resume: bool
    replace_output_head: bool  # replace output head when resume
    checkpoint: Optional[Dict[str, Any]]

    # dataset feature
    original_feature_stats: FeatureStats
    merged_feature_stats: FeatureStats
    task_type: TaskType

    # learning rate
    transformer_lr: float
    output_head_lr: float
    lr_scheduler: Literal['constant', 'cosine']
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float

    # loss_type
    loss_type: Literal['BINCE', 'MULCE', 'MSE', 'SUPCON']

    # random generator
    train_rng: random.Random
    loss_rng: random.Random

    def __init__(self, hp: HyperParameters, ts: TrainSettings):
        assert isinstance(hp, HyperParameters)
        assert isinstance(ts, TrainSettings)
        assert all('cuda' not in device
                   for device in [ts.device, ts.dataset_device]) \
            or torch.cuda.is_available(), "cuda is not available"
        assert 'cuda' in ts.device or ts.dtype != 'bfloat16', 'only cuda support bfloat16 dtype'
        self.hp = hp
        self.ts = ts

    def train(self,
              data_reader: DataReader,
              tp: TrainParameters,
              resume: bool = False,
              replace_output_head: Optional[bool] = None
              ):

        assert isinstance(data_reader, DataReader)
        assert isinstance(tp, TrainParameters)
        assert isinstance(resume, bool)
        assert replace_output_head is None or isinstance(
            replace_output_head, bool)

        self.data_reader = data_reader
        self.tp = tp

        self.output_checkpoint = tp.output_checkpoint \
            if tp.output_checkpoint is not None else tp.checkpoint
        assert self.output_checkpoint is not None

        self.input_checkpoint = tp.input_checkpoint \
            if tp.input_checkpoint is not None else tp.checkpoint
        assert self.input_checkpoint is not None

        assert replace_output_head is None or \
            not replace_output_head or resume, \
            "when `replace_output_head` is True, `resume` must be True"

        self.resume = resume
        self.replace_output_head = replace_output_head

        self.transformer_lr = self.tp.transformer_lr \
            if self.tp.transformer_lr is not None else self.tp.learning_rate

        self.output_head_lr = self.tp.output_head_lr \
            if self.tp.output_head_lr is not None else self.tp.learning_rate

        assert self.transformer_lr is not None and self.output_head_lr is not None

        self.lr_scheduler = self.tp.lr_scheduler
        assert self.lr_scheduler in ('constant', 'cosine')

        assert self.tp.loss_type != 'SUPCON' or self.tp.output_dim % 16 == 0, \
            "`output_dim` must be multiple of 16 for `SUPCON` loss type."

        self.train_rng = random.Random(self.ts.dataset_seed + 138321)
        self.loss_rng = random.Random(self.ts.dataset_seed + 140987)

        self.loss_type = self.tp.loss_type

        self._init_torch()

        if self.resume:
            self._load_checkpoint()

        self._create_dataset()

        self._create_model()

        if self.resume:
            self._load_state()

        if self.replace_output_head:
            self.reset_output_head()

        self._init_dataloader()

        self._train()

        # after train, free up resources to reduce video memory usage
        self._freeup()

    def _train(self):

        config = self.hp.asdict() | self.ts.asdict() | self.tp.asdict()

        os.makedirs(self.ts.out_dir, exist_ok=True)

        # for later use in torch.autocast
        self.device_type = "cuda" if "cuda" in self.ts.device else "cpu"

        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32,
                   "bfloat16": torch.bfloat16, "float16": torch.float16}[self.ts.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        )

        # load model to device
        self.model.to(self.ts.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        # scaler = torch.cuda.amp.GradScaler(
        #     enabled=(self.ts.dtype == "float16"))
        scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.ts.dtype == "float16"))
        # optimizer
        self.optimizer = self._configure_optimizers(
            self.hp.weight_decay, (self.hp.beta1, self.hp.beta2), self.device_type)

        if self.resume and not self.replace_output_head and "optimizer" in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])

        # free up resource
        self.checkpoint = None

        # compile the model
        if self.ts.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        # logging
        if self.ts.wandb_log:
            import wandb
            datestr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            run_name = self.ts.wandb_run_name + '_' + datestr
            wandb.init(project=self.ts.wandb_project,
                       name=run_name, config=config)

        self.lr_decay_iters = self.tp.max_iters
        self.min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        self.warmup_iters = self.tp.warmup_iters
        assert self.warmup_iters < 0.3 * \
            self.lr_decay_iters, "warmup_iters too large compared to train_epochs."

        iter_num = 0
        best_val_loss = 1e9
        t0 = time.time()
        running_mfu = -1.0

        batch_seed = self.train_rng.randint(1024, 1024*1024)
        train_batch_iter = self.iter_batches(
            split="train", seed=batch_seed)

        iters_per_epoch = self.dataset.n_train // self.tp.batch_size
        # iterate over the dataset
        for X, Y in train_batch_iter:
            current_lr = {}
            # determine and set the learning rate for this iteration
            for name, param_group in zip(self.optim_group_names, self.optimizer.param_groups):
                lr = self.get_lr(iter_num, name)
                param_group["lr"] = lr
                current_lr.update({f"lr/{name}": lr})

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.tp.eval_interval == 0:
                losses = self._estimate_loss()
                print(f"""step {iter_num}: train loss {
                    losses['train']:.4f}, val loss {losses['val']:.4f}""")
                if self.ts.wandb_log:
                    self._log(wandb, iter_num, losses,
                              current_lr, running_mfu)
                if losses["val"] < best_val_loss or self.tp.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        self._save_checkpoint(
                            iter_num, best_val_loss, config)
            if iter_num == 0 and self.ts.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            with self.ctx:
                logits = self.model(X, Y)
                loss = self.model.last_loss

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

            # clip the gradient
            if self.hp.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.hp.grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(self.optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.ts.log_interval == 0:
                # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item()
                if iter_num >= 5:  # let the training loop settle a bit
                    mfu = self._estimate_mfu(
                        self.tp.batch_size, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                lr_str = f"lr {list(current_lr.values())[0]:e}" \
                    if len(set(current_lr.values())) == 1 \
                    else f"lr/t {current_lr['lr/transformer']:e} | lr/o {current_lr['lr/output']:e}"
                epochs = iter_num / iters_per_epoch
                print(
                    f"{iter_num} | epoch {epochs:.4f} | loss {lossf:.4f} |"
                    f"{lr_str} |{dt*1000: .2f}ms | mfu {running_mfu*100: .2f}%"
                )

            iter_num += 1

            if iter_num > self.tp.max_iters:
                break

    def _log(self, wandb, iter_num, losses, lr, running_mfu):
        try:
            log_dict = {
                "iter": iter_num,
                "loss/train": losses["train"],
                "loss/val": losses["val"],
                # "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }
            log_dict.update(lr)
            wandb.log(log_dict, step=iter_num)
        except Exception as e:
            print(f"logging to wandb failed: {e}")

    def _save_checkpoint(self, iter_num, best_val_loss, config):
        save_checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_args": self.model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
            "features": {"feature_stats": self.merged_feature_stats,
                         "task_type": self.task_type},
        }
        print(f"saving checkpoint to {self.ts.out_dir}")
        torch.save(save_checkpoint, os.path.join(
            self.ts.out_dir, self.output_checkpoint))

    def _init_torch(self):
        # init seed before create model
        torch.manual_seed(self.ts.torch_seed)
        # when enabled, pyTorch is allowed to use the TensorFloat32 (TF32) tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    def _create_model(self):

        self.model_args = ModelArgs(
            dim=self.hp.dim,
            n_layers=self.hp.n_layers,
            n_heads=self.hp.n_heads,
            loss_type=self.tp.loss_type,
            feature_vocab_size=self.merged_feature_stats.vocab_size,
            output_dim=self.tp.output_dim,
            output_hidden_dim=self.hp.output_hidden_dim,
            output_forward_dim=self.hp.output_forward_dim,
            multiple_of=self.hp.multiple_of,
            max_seq_len=self.merged_feature_stats.seq_len,
            dropout=self.hp.dropout,
        )
        model_args_dict = self.model_args.asdict()

        if self.resume:
            checkpoint_model_args: ModelArgs = self.checkpoint["model_args"]
            equal, diff = equals_except(
                model_args_dict,
                checkpoint_model_args.asdict(),
                ['loss_type', 'output_dim', 'dropout'])
            if not equal:
                raise ValueError(
                    f'model_args not consistent with checkpoint: {diff}')
            model_args_dict['output_dim'] = checkpoint_model_args.output_dim

        self.model = TabularTransformer(ModelArgs(**model_args_dict))

    def reset_output_head(self):
        self.model.reset_output_head(self.model_args.output_dim)

    def _load_state(self):
        state_dict = self.checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            # load state dict
        self.model.load_state_dict(state_dict)

    def _create_dataset(self):
        if self.resume:
            features = copy.deepcopy(self.checkpoint['features'])
            self.original_feature_stats = features['feature_stats']
            if self.replace_output_head:
                self.original_feature_stats = self.original_feature_stats.reset_y_stats()
        else:
            self.original_feature_stats = None

        self.dataset = TabularDataset(
            datareader=self.data_reader,
            device=self.ts.dataset_device,
            original_feature_stats=self.original_feature_stats,
            min_cat_count=self.ts.min_cat_count,
            apply_power_transform=self.ts.apply_power_transform,
            validate_split=self.tp.validate_split,
            seed=self.ts.dataset_seed
        )

        self.task_type = self.dataset.task_type
        self.merged_feature_stats = self.dataset.merged_feature_stats

        assert self.task_type is not TaskType.BINCLASS or \
            self.loss_type in ('BINCE', 'SUPCON'), \
            "only binary cross entropy loss or supervised contrastive loss could be used for binary classification task"

        assert self.task_type is not TaskType.MULTICLASS or \
            self.loss_type in ('MULCE', 'SUPCON'), \
            "only multi class cross entropy loss or supervised contrastive loss could be used for multi classification task"

        assert self.task_type is not TaskType.REGRESSION or \
            self.loss_type in ('MSE',), \
            "only MSE loss could be used for regression task"

        assert LossType[self.loss_type] is LossType.SUPCON or \
            self.dataset.n_class == self.tp.output_dim,  \
            f"dataset target has `n_class` {self.dataset.n_class}, " \
            f"but given `output_dim` {self.tp.output_dim}"

    def _load_checkpoint(self):
        ckpt_path = os.path.join(
            self.ts.out_dir, self.input_checkpoint)
        self.checkpoint = torch.load(
            ckpt_path,
            map_location=self.ts.device,
            weights_only=False
        )

    def _init_dataloader(self):
        # task-specific setup
        self.iter_batches = partial(
            Task.iter_batches,
            batch_size=self.tp.batch_size,
            tabular_dataset=self.dataset,
            unk_ratio=self.tp.unk_ratio,
            unk_ratio_default=self.ts.unk_ratio_default,
            device=self.ts.device,
        )

    def _freeup(self):
        self.model = None
        self.optimizer = None
        torch.cuda.empty_cache()

    def _estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.model.parameters())
        cfg = self.model.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def _configure_optimizers(self, weight_decay, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        num_params = sum(p.numel() for pn, p in param_dict.items())
        print("num parameter tensors: "
              f"{len(param_dict)}, with {num_params:,} parameters")

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        transformer_decay_params = [
            p for n, p in param_dict.items() if p.dim() >= 2 and not n.startswith('output')]
        output_decay_params = [
            p for n, p in param_dict.items() if p.dim() >= 2 and n.startswith('output')]
        transformer_nodecay_params = [
            p for n, p in param_dict.items() if p.dim() < 2 and not n.startswith('output')]
        output_nodecay_params = [
            p for n, p in param_dict.items() if p.dim() < 2 and n.startswith('output')]
        self.optim_group_names = ['transformer',
                                  'transformer', 'output', 'output']
        optim_groups = [
            {'params': transformer_decay_params,
                'weight_decay': weight_decay, 'lr': self.transformer_lr},
            {'params': transformer_nodecay_params,
                'weight_decay': 0.0, 'lr': self.transformer_lr},
            {'params': output_decay_params,
                'weight_decay': weight_decay, 'lr': self.output_head_lr},
            {'params': output_nodecay_params,
                'weight_decay': 0.0, 'lr': self.output_head_lr},
        ]
        transformer_num_decay_params = sum(
            p.numel() for p in transformer_decay_params)
        output_num_decay_params = sum(p.numel() for p in output_decay_params)
        transformer_num_nodecay_params = sum(
            p.numel() for p in transformer_nodecay_params)
        output_num_nodecay_params = sum(p.numel()
                                        for p in output_nodecay_params)

        print("Transformer num decayed parameter tensors: "
              f"{len(transformer_decay_params)}, with {transformer_num_decay_params:,} parameters")
        print("Transformer num non-decayed parameter tensors: "
              f"{len(transformer_nodecay_params)}, with {transformer_num_nodecay_params:,} parameters")
        print("Output num decayed parameter tensors: "
              f"{len(output_decay_params)}, with {output_num_decay_params:,} parameters")
        print("Output num non-decayed parameter tensors: "
              f"{len(output_nodecay_params)}, with {output_num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def _estimate_loss(self):
        out = {}
        eval_iters = self.tp.eval_iters
        self.model.eval()
        split_arr = ["train", "val"] \
            if self.dataset.n_validate > 0 else ["train"]
        for split in split_arr:
            k = 0
            losses = torch.zeros(eval_iters)  # keep on CPU
            while (k < eval_iters):
                seed = self.loss_rng.randint(1024, 1024*1024)
                batch_iter = self.iter_batches(
                    split=split, seed=seed)
                for X, Y in batch_iter:
                    with self.ctx:
                        logits = self.model(X, Y)
                        loss = self.model.last_loss
                    losses[k] = loss.item()
                    k += 1
                    if k >= eval_iters:
                        break
            out[split] = losses.mean()
        out.setdefault('val', torch.tensor(0.))
        self.model.train()
        return out

    def get_lr(self, it: int, model_part: Literal['transformer', 'output']):
        assert model_part in ('transformer', 'output')

        learning_rate = self.transformer_lr if model_part == 'transformer' else self.output_head_lr

        if self.lr_scheduler == 'constant':
            return learning_rate

        # cosine learning rate schedule
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / \
            (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (learning_rate - self.min_lr)
