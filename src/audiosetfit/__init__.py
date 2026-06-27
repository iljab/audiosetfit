"""audiosetfit: efficient few-shot audio classification with contrastive fine-tuning.

The public API intentionally mirrors Hugging Face SetFit so that knowledge transfers
1:1 from text to audio:

    from audiosetfit import AudioSetFitModel, Trainer, TrainingArguments, sample_dataset
"""

from .data import sample_dataset
from .encoders import (
    ASTEncoder,
    AudioEncoder,
    ClapAudioEncoder,
    Wav2Vec2LikeEncoder,
    WhisperEncoder,
    build_encoder,
)
from .losses import ContrastiveLoss, CosineSimilarityLoss, SupConLoss, get_loss
from .modeling import AudioSetFitHead, AudioSetFitModel
from .sampler import ContrastiveDataset
from .trainer import Trainer
from .training_args import TrainingArguments

__version__ = "0.1.0"

__all__ = [
    "AudioSetFitModel",
    "AudioSetFitHead",
    "Trainer",
    "TrainingArguments",
    "sample_dataset",
    "ContrastiveDataset",
    "AudioEncoder",
    "ClapAudioEncoder",
    "ASTEncoder",
    "Wav2Vec2LikeEncoder",
    "WhisperEncoder",
    "build_encoder",
    "CosineSimilarityLoss",
    "ContrastiveLoss",
    "SupConLoss",
    "get_loss",
]
