from typing import Literal, Optional
from .hyperparameters import HyperParameters
from .data_common import DataReader
import torch
import inspect


class Trainer:
    def __init__(self, data_reader: DataReader, hp: HyperParameters):
        assert isinstance(data_reader, DataReader)
        assert isinstance(hp, HyperParameters)
        self.data_reader = data_reader
        self.hp = hp
        self.model = None

    def train(resume: bool = False,
              finetue: bool = False,
              input_checkpoint: Optional[str] = None,
              output_checkpoint: str = 'ckpt.pt'):
        ...

    def _make_model():
        ...

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        num_params = sum(p.numel() for pn, p in param_dict.items())
        print("num parameter tensors: "
              f"{len(param_dict)}, with {num_params:,} parameters")
        if self.finetune:
            print("finetune mode pretrained parameters learning rate scaled by 0.1")
            finetune_params = [
                p for n, p in param_dict.items() if not n.startswith('output')]
            output_params = [p for n, p in param_dict.items()
                             if n.startswith('output')]
            optim_groups = [
                {'params': finetune_params, 'lr': 0.0},
                # {'params': finetune_params, 'lr': learning_rate},
                {'params': output_params}
            ]

        else:
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print("num decayed parameter tensors: "
                  f"{len(decay_params)}, with {num_decay_params:,} parameters")
            print("num non-decayed parameter tensors: "
                  f"{len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    # @torch.no_grad()
    # def estimate_loss(self):
    #     out = {}
    #     model.eval()
    #     for split in ["train", "val"]:
    #         k = 0
    #         losses = torch.zeros(hp.eval_iters)  # keep on CPU
    #         while (k < hp.eval_iters):
    #             seed = loss_estimate_rng.randint(1024, 1024*1024)
    #             batch_iter = iter_batches(
    #                 split=split, seed=seed, unk_ratio=hp.unk_ratio)
    #             for X, Y in batch_iter:
    #                 with ctx:
    #                     logits = model(X, Y)
    #                     loss = raw_model.last_loss
    #                 losses[k] = loss.item()
    #                 k += 1
    #                 if k >= hp.eval_iters:
    #                     break
    #         out[split] = losses.mean()
    #     model.train()
    #     return out

    # def get_lr(self, it):
    #     # finetune
    #     if hp.finetune:
    #         return hp.learning_rate
    #     # 1) linear warmup for warmup_iters steps
    #     if it < hp.warmup_iters:
    #         return hp.learning_rate * it / hp.warmup_iters
    #     # 2) if it > lr_decay_iters, return min learning rate
    #     if it > lr_decay_iters:
    #         return min_lr
    #     # 3) in between, use cosine decay down to min learning rate
    #     decay_ratio = (it - hp.warmup_iters) / \
    #         (lr_decay_iters - hp.warmup_iters)
    #     assert 0 <= decay_ratio <= 1
    #     # coeff ranges 0..1
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    #     return min_lr + coeff * (hp.learning_rate - min_lr)
