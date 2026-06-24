"""Contrastive losses operating directly on embedding tensors.

Reimplementing the small amount of loss math needed for the
embedding fine-tuning phase. Each loss takes two batches of embeddings and a float
label (1.0 = same class, 0.0 = different class).
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


_LOSSES = {
    "cosine": CosineSimilarityLoss,
    "contrastive": ContrastiveLoss,
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
