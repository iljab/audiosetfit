"""Training arguments for the two-phase audiosetfit `Trainer`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from torch import nn


@dataclass
class TrainingArguments:
    """Configuration for both training phases.

    Phase 1 (embedding fine-tuning, contrastive):
        embedding_num_epochs, embedding_batch_size, body_learning_rate, loss, margin,
        sampling_strategy, num_iterations, max_steps, max_pairs, warmup_proportion, l2_weight.

    Phase 2 (classifier head):
        classifier_num_epochs, classifier_batch_size, head_learning_rate, end_to_end.

    Set ``train_embeddings=False`` to skip phase 1 and use the frozen backbone directly
    (often a strong baseline with an already-contrastive backbone like CLAP).
    """

    output_dir: str = "audiosetfit-output"
    seed: int = 42

    # ---- phase 1: embedding fine-tuning ----
    train_embeddings: bool = True
    embedding_num_epochs: int = 1
    embedding_batch_size: int = 16
    body_learning_rate: float = 2e-5
    loss: Union[str, nn.Module, type] = "cosine"
    margin: float = 0.5
    sampling_strategy: str = "oversampling"
    num_iterations: Optional[int] = None
    max_steps: int = -1
    max_pairs: int = -1
    warmup_proportion: float = 0.1
    l2_weight: float = 0.01
    use_amp: bool = False

    # ---- phase 2: classifier head ----
    classifier_num_epochs: int = 25
    classifier_batch_size: int = 16
    head_learning_rate: float = 1e-2
    end_to_end: bool = False

    # ---- misc ----
    show_progress_bar: bool = True
    num_workers: int = 0

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sampling_strategy not in {"unique", "oversampling", "undersampling"}:
            raise ValueError(
                f"sampling_strategy must be one of 'unique', 'oversampling', 'undersampling', "
                f"got {self.sampling_strategy!r}."
            )
