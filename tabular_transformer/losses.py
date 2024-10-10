import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, pred_scores, true_ranks):
        """
        Calculate pairwise rank loss between predicted scores and true ranks.

        Args:
            pred_scores: Tensor of predicted scores (1D tensor).
            true_ranks: Tensor of true ranks (1D tensor with rank values).

        Returns:
            loss: Computed loss value.
        """

        assert pred_scores.dim() == 1, "`pred_scores` must be a 1D tensor"
        assert true_ranks.dim() == 1, "`true_ranks` must be a 1D tensor"
        assert pred_scores.size(0) == true_ranks.size(0), \
            "`pred_scores` and `true_ranks` must have the same length"

        # Compute pairwise score differences
        predict_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)

        # Compute pairwise rank differences
        rank_diff = torch.sign(
            true_ranks.unsqueeze(0) - true_ranks.unsqueeze(1))

        # Convert rank differences to binary format for BCE (0 or 1)
        truth = 0.5 * (rank_diff + 1.0)

        # Compute loss using BCEWithLogitsLoss
        loss = F.binary_cross_entropy_with_logits(predict_diff, truth)
        return loss
