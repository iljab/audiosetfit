"""Contrastive pair generation for the embedding fine-tuning phase.

Ported from SetFit's `sampler.py`. The key property is that pairing only depends on
**labels**, so the exact same logic works for audio. To avoid duplicating (potentially
large) waveforms, pairs are stored as *index pairs* into the example list; the actual
audio is loaded lazily by the trainer's collate function.
"""

from __future__ import annotations

from itertools import zip_longest
from typing import Dict, Generator, Iterable, List, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset


def shuffle_combinations(iterable: Iterable, replacement: bool = True) -> Generator:
    """Generate deterministically shuffled pair combinations (by index) for an iterable."""
    n = len(iterable)
    k = 1 if not replacement else 0
    idxs = np.stack(np.triu_indices(n, k), axis=-1)
    for i in np.random.RandomState(seed=42).permutation(len(idxs)):
        _idx, idx = idxs[i, :]
        yield iterable[_idx], iterable[idx]


class ContrastiveDataset(IterableDataset):
    """Generates positive (same-label) and negative (different-label) index pairs.

    Args:
        labels: per-example labels (ints, or multi-hot lists when ``multilabel=True``).
        multilabel: treat labels as multi-hot vectors.
        num_iterations: if set, total pairs = ``num_iterations * n_examples`` per polarity.
        sampling_strategy: one of ``"unique"``, ``"oversampling"``, ``"undersampling"``.
        max_pairs: hard cap on total generated pairs (``-1`` = no cap).

    Each yielded item is ``{"idx_1": int, "idx_2": int, "label": float}`` where the
    label is ``1.0`` for positive pairs and ``0.0`` for negative pairs.
    """

    def __init__(
        self,
        labels: List[Union[int, List[int]]],
        multilabel: bool = False,
        num_iterations: Optional[int] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        super().__init__()
        self.pos_index = 0
        self.neg_index = 0
        self.pos_pairs: List[Dict[str, Union[int, float]]] = []
        self.neg_pairs: List[Dict[str, Union[int, float]]] = []
        self.labels = labels
        self.indexed_labels = list(enumerate(labels))
        self.max_pos_or_neg = -1 if max_pairs == -1 else max_pairs // 2

        if multilabel:
            self.generate_multilabel_pairs()
        else:
            self.generate_pairs()

        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.labels)
            self.len_neg_pairs = num_iterations * len(self.labels)
        elif sampling_strategy == "unique":
            self.len_pos_pairs = len(self.pos_pairs)
            self.len_neg_pairs = len(self.neg_pairs)
        elif sampling_strategy == "undersampling":
            self.len_pos_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
        elif sampling_strategy == "oversampling":
            self.len_pos_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
        else:
            raise ValueError(
                "Invalid sampling strategy. Must be one of 'unique', 'oversampling', or 'undersampling'."
            )

    def _append(self, idx_1: int, label_1, idx_2: int, label_2, is_positive: bool) -> bool:
        pos_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
        neg_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg
        if is_positive:
            if not pos_full:
                self.pos_pairs.append({"idx_1": idx_1, "idx_2": idx_2, "label": 1.0})
        elif not neg_full:
            self.neg_pairs.append({"idx_1": idx_1, "idx_2": idx_2, "label": 0.0})
        return pos_full and neg_full

    def generate_pairs(self) -> None:
        for (idx_1, label_1), (idx_2, label_2) in shuffle_combinations(self.indexed_labels):
            if self._append(idx_1, label_1, idx_2, label_2, is_positive=label_1 == label_2):
                break

    def generate_multilabel_pairs(self) -> None:
        for (idx_1, label_1), (idx_2, label_2) in shuffle_combinations(self.indexed_labels):
            is_positive = bool(np.any(np.logical_and(label_1, label_2)))
            if self._append(idx_1, label_1, idx_2, label_2, is_positive=is_positive):
                break

    def get_positive_pairs(self) -> List[Dict[str, Union[int, float]]]:
        pairs = []
        for _ in range(self.len_pos_pairs):
            if self.pos_index >= len(self.pos_pairs):
                self.pos_index = 0
            pairs.append(self.pos_pairs[self.pos_index])
            self.pos_index += 1
        return pairs

    def get_negative_pairs(self) -> List[Dict[str, Union[int, float]]]:
        pairs = []
        for _ in range(self.len_neg_pairs):
            if self.neg_index >= len(self.neg_pairs):
                self.neg_index = 0
            pairs.append(self.neg_pairs[self.neg_index])
            self.neg_index += 1
        return pairs

    def __iter__(self):
        for pos_pair, neg_pair in zip_longest(self.get_positive_pairs(), self.get_negative_pairs()):
            if pos_pair is not None:
                yield pos_pair
            if neg_pair is not None:
                yield neg_pair

    def __len__(self) -> int:
        return self.len_pos_pairs + self.len_neg_pairs
