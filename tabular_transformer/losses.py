import torch
import torch.nn as nn
import math


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.2):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].

        Returns:
            A loss scalar.
        """

        device = features.device

        assert len(
            features.shape) == 2, "`features` needs to be [bsz, feature_dim]"
        assert len(labels.shape) == 1, "`labels` needs to be shape [bsz]"

        assert features.shape[0] == labels.shape[0]
        batch_size, _ = features.shape

        # normalize the feature
        features = features / torch.linalg.norm(features, dim=1, keepdim=True)

        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.matmul(features, features.T)
        logits = torch.div(anchor_dot_contrast, self.temperature)
        logits_mask = 1.0 - \
            torch.eye(batch_size, dtype=features.dtype, device=device)

        mask = mask * logits_mask

        log_prob_pos = logsumexp(logits, mask, keepdim=False)
        log_prob_all = logsumexp(logits, logits_mask, keepdim=False)

        log_prob = log_prob_pos - log_prob_all

        n_pos_pairs = (mask.sum(1) >= 1).float()

        # loss
        mean_log_prob_pos = -(log_prob * n_pos_pairs)
        loss = mean_log_prob_pos.sum() / (n_pos_pairs.sum() + 1e-8)

        assert not torch.isnan(loss).any().item()
        # The loss gradient naturally has a 1/temperature scaling factor, so this
        # counteracts it.
        # loss *= self.temperature
        return loss


def logsumexp(x, logits_mask, dim=1, keepdim=True):
    x = x.masked_fill(~logits_mask.bool(), torch.finfo(x.dtype).min)
    output = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    return output
