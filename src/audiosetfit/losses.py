"""Contrastive losses operating directly on embedding tensors.

Reimplementing the small amount of loss math needed for the
embedding fine-tuning phase. Two families live here:

* **Pairwise** losses (``CosineSimilarityLoss``, ``ContrastiveLoss``) take two batches of
  embeddings and a float label (1.0 = same class, 0.0 = different class).
* **In-batch** losses (``SupConLoss``) take a single batch of embeddings plus integer class
  labels and use every other same-class example in the batch as a positive and the rest as
  negatives. These are marked with the class attribute ``in_batch = True`` so the ``Trainer``
  knows to feed them grouped batches instead of pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CosineSimilarityLoss(nn.Module):
    """SetFit's default. Pull positive pairs toward cosine 1.0, negatives toward 0.0.

    loss = MSE(cos_sim(u, v), label)
    """

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return F.mse_loss(cos, labels.to(cos.dtype))


class ContrastiveLoss(nn.Module):
    """Classic contrastive (Hadsell et al.) loss on (normalized) Euclidean distance.

    loss = label * d^2 + (1 - label) * relu(margin - d)^2
    """

    def __init__(self, margin: float = 0.5, normalize: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            emb_a = F.normalize(emb_a, p=2, dim=-1)
            emb_b = F.normalize(emb_b, p=2, dim=-1)
        distances = F.pairwise_distance(emb_a, emb_b, p=2)
        labels = labels.to(distances.dtype)
        losses = labels * distances.pow(2) + (1 - labels) * F.relu(self.margin - distances).pow(2)
        return losses.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020) with in-batch negatives.

    Operates on a single batch of embeddings ``[B, D]`` and integer class labels ``[B]``.
    For each anchor, every *other* same-class sample in the batch is a positive and all
    remaining samples are negatives, so a batch of size B yields up to B*(B-1) implicit
    comparisons (unlike the pairwise losses, larger batches add real negatives here).

    An anchor only contributes if its class appears at least twice in the batch, so the
    ``Trainer`` pairs this loss with a group-by-label batch sampler.
    """

    in_batch = True  # tells the Trainer to use the grouped-batch (non-pair) path

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(embeddings, p=2, dim=-1)
        sim = feats @ feats.t() / self.temperature
        # Numerical stability (subtract per-row max; detached so it doesn't affect grads).
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        eye = torch.eye(sim.size(0), device=feats.device)
        pos_mask = (labels == labels.t()).float() - eye  # same-class, excluding self
        logits_mask = 1.0 - eye  # exclude self from the denominator

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0
        if valid.sum() == 0:
            # No in-batch positives: return a graph-connected zero so .backward() is safe.
            return feats.sum() * 0.0
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[valid]
        return -mean_log_prob_pos.mean()


_LOSSES = {
    "cosine": CosineSimilarityLoss,
    "contrastive": ContrastiveLoss,
    "supcon": SupConLoss,
    "supervised_contrastive": SupConLoss,
}


def get_loss(loss: "str | nn.Module | type") -> nn.Module:
    """Resolve a loss given a name, an instance, or a class."""
    if isinstance(loss, nn.Module):
        return loss
    if isinstance(loss, type) and issubclass(loss, nn.Module):
        return loss()
    if isinstance(loss, str):
        key = loss.lower()
        if key not in _LOSSES:
            raise ValueError(f"Unknown loss '{loss}'. Available: {sorted(_LOSSES)}.")
        return _LOSSES[key]()
    raise TypeError(f"Unsupported loss specification: {loss!r}")
