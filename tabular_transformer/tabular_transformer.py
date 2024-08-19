import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tabular_transformer.util import LossType
from tabular_transformer.losses import SupConLoss
from tabular_transformer.hyperparameters import ModelArgs


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            # manual implementation
            # (bs, n_heads, seqlen, seqlen)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / \
                math.sqrt(self.head_dim)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # (bs, n_heads, seqlen, head_dim)
            output = torch.matmul(scores, xv)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * \
                ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x):
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class NumericValueEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (batch_size, seq_len) with values in range (0, 1)
        """
        batch_size, seq_len = x.size()
        numeric_val = x.unsqueeze(-1)  # Shape (batch_size, seq_len, 1)
        mul_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=x.dtype,
                             device=x.device) * (math.log(10000.0) / self.embedding_dim))

        numeric_val_enc = torch.zeros(
            batch_size, seq_len, self.embedding_dim, device=x.device)
        numeric_val_enc[:, :, 0::2] = torch.sin(numeric_val * mul_term)
        numeric_val_enc[:, :, 1::2] = torch.cos(numeric_val * mul_term)

        return numeric_val_enc


class Transformer(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.feature_vocab_size = params.feature_vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.feature_vocab_size, params.dim)
        self.num_embeddings = NumericValueEncoding(params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        feature_tokens: the index for token embedding, (bsz, seqlen) equals (rows, colums)
        feature_weights: for categorical variable the value is 1.0, for numerical variable the value is normalized value, same size as feature_tokens
        """
        feature_tokens, feature_weight = features
        he = self.tok_embeddings(feature_tokens)
        ve = self.num_embeddings(feature_weight)
        # experiments show that multiplication is slightly better than addition
        # but keep it simple according to Occam's razor rule
        # hw = he * ve
        hw = he + ve
        h = self.dropout(hw)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        return h


class ForwardOutPut(nn.Module):
    def __init__(self, params: ModelArgs, output_dim: Optional[int] = None, bias: bool = False):
        super().__init__()

        self.hidden_dim = params.output_hidden_dim
        self.forward_dim = params.output_forward_dim
        self.dim = params.dim
        self.output_dim = output_dim if output_dim is not None else params.output_dim
        assert self.output_dim > 0, "`output_dim` must be larger than 0"
        self.hidden_cat_dim = params.max_seq_len * self.forward_dim

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        self.w2 = nn.Linear(self.hidden_dim, self.forward_dim, bias=bias)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=bias)

        self.wh = nn.Linear(self.hidden_cat_dim, self.hidden_dim, bias=bias)
        self.wo = nn.Linear(self.hidden_dim, self.output_dim, bias=bias)

        self.dropout = nn.Dropout(params.dropout)
        self.out_norm = RMSNorm(self.hidden_cat_dim, eps=1e-5)

    def forward(self, x):
        forward_output = self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        bsz = forward_output.shape[0]
        forward_output = forward_output.view(bsz, -1)

        hidden_output = self.wh(self.out_norm(forward_output))
        output = self.wo(F.silu(hidden_output))
        return output


class TabularTransformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        assert params.loss_type in ('BINCE', 'MULCE', 'MSE', 'SUPCON')
        self.loss_type = LossType[params.loss_type]
        self.transformer = Transformer(params)
        self.output = ForwardOutPut(params)
        self.sup_con_loss = SupConLoss() if self.loss_type is LossType.SUPCON else None
        # self.finetune = params.finetune
        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_output_head(self, output_dim: int):
        self.output = ForwardOutPut(self.params, output_dim)
        self.output.apply(self._init_weights)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor], targets: Optional[torch.Tensor] = None) -> torch.Tensor:

        h = self.transformer(features)

        logits: torch.Tensor = self.output(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss

            if self.loss_type is LossType.BINCE:
                assert logits.size(-1) == 1
                self.last_loss = F.binary_cross_entropy_with_logits(
                    # 0 - 1 classification
                    logits.squeeze(-1), targets.float())
            elif self.loss_type is LossType.MULCE:
                self.last_loss = F.cross_entropy(
                    logits, targets)  # multi-class
            elif self.loss_type is LossType.MSE:
                assert logits.size(-1) == 1
                self.last_loss = F.mse_loss(
                    logits.squeeze(-1), targets)  # (x_n - y_n)**2
            elif self.loss_type is LossType.SUPCON:
                self.last_loss = self.sup_con_loss(logits, targets)
            else:
                raise ValueError("unknown loss function type.")
        else:
            self.last_loss = None

        return logits

    @torch.inference_mode()
    def predict(self, features: Tuple[torch.Tensor, torch.Tensor], targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits: torch.Tensor = self(features, targets)
        return logits, self.last_loss
