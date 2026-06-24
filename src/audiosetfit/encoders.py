"""Audio encoders: turn waveforms into fixed-size embeddings.

`AudioEncoder` is the body of an `AudioSetFitModel`, analogous to a `SentenceTransformer`
in SetFit. It is an `nn.Module` so it can be fine-tuned during the contrastive phase and
moved across devices.

Shipped backbones:
  * `ClapAudioEncoder` (default) — LAION CLAP (auto-detected: model_type "clap").
  * `ASTEncoder` — Audio Spectrogram Transformer.
  * `Wav2Vec2LikeEncoder` — wav2vec2 / HuBERT / WavLM and friends (masked mean pooling).
  * `WhisperEncoder` — Whisper encoder tower.

Adding another backbone is a small change: subclass `AudioEncoder`, implement `prepare`
+ `forward_features`, and register it in `_ENCODER_REGISTRY`. See the README for an example.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .data import AudioInput, load_audio_batch


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AudioEncoder(nn.Module):
    """Abstract base class for audio embedding bodies.

    Subclasses must set ``self.target_sr`` and ``self.embedding_dim`` and implement
    ``prepare`` (waveforms -> model inputs) and ``forward_features`` (inputs -> [B, D]).
    """

    model_id: str
    target_sr: int
    embedding_dim: int

    def prepare(self, waveforms: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed(self, waveforms: List[np.ndarray]) -> torch.Tensor:
        """Differentiable: waveforms -> embeddings [B, D] (used during training)."""
        inputs = self.prepare(waveforms)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.forward_features(inputs)

    @torch.no_grad()
    def encode(
        self,
        inputs: List[AudioInput],
        batch_size: int = 16,
        normalize: bool = True,
        convert_to_tensor: bool = False,
        show_progress_bar: Optional[bool] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Non-differentiable batched encoding (used for head fitting and inference)."""
        self.eval()
        waveforms = load_audio_batch(inputs, self.target_sr)
        all_embeds: List[torch.Tensor] = []
        iterator = range(0, len(waveforms), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding", leave=False)
        for start in iterator:
            batch = waveforms[start : start + batch_size]
            embeds = self.embed(batch)
            if normalize:
                embeds = nn.functional.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.detach().cpu())
        embeds = torch.cat(all_embeds, dim=0)
        return embeds if convert_to_tensor else embeds.numpy()

    def save(self, save_directory: str) -> None:
        raise NotImplementedError


class ClapAudioEncoder(AudioEncoder):
    """LAION CLAP audio tower + projection head as a fixed-size embedder (512-d)."""

    def __init__(self, model_id: str = "laion/clap-htsat-unfused", device: Optional[str] = None) -> None:
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        self.model_id = model_id
        self.clap = ClapModel.from_pretrained(model_id)
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.target_sr = int(self.processor.feature_extractor.sampling_rate)
        self.embedding_dim = int(self.clap.config.projection_dim)
        self.to(_resolve_device(device))

    def prepare(self, waveforms: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        audio = [np.asarray(w, dtype=np.float32) for w in waveforms]
        # transformers >= 5 renamed the `audios` kwarg to `audio`.
        try:
            return self.processor(audio=audio, sampling_rate=self.target_sr, return_tensors="pt")
        except (TypeError, ValueError):
            return self.processor(audios=audio, sampling_rate=self.target_sr, return_tensors="pt")

    def forward_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.clap.get_audio_features(**inputs)
        # transformers >= 5 returns a BaseModelOutputWithPooling whose `pooler_output`
        # is the (already L2-normalized) audio embedding; older versions return a tensor.
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        return out[0]

    def save(self, save_directory: str) -> None:
        self.clap.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)


class ASTEncoder(AudioEncoder):
    """Audio Spectrogram Transformer (AST) as a fixed-size embedder.

    Uses AST's pooled output (mean of the CLS + distillation tokens) when available,
    otherwise mean-pools the last hidden state. Great for AudioSet-style tagging.
    """

    def __init__(self, model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593", device: Optional[str] = None) -> None:
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_id = model_id
        self.backbone = AutoModel.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.target_sr = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.embedding_dim = int(self.backbone.config.hidden_size)
        self.to(_resolve_device(device))

    def prepare(self, waveforms: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        return self.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )

    def forward_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return outputs.last_hidden_state.mean(dim=1)

    def save(self, save_directory: str) -> None:
        self.backbone.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)


class Wav2Vec2LikeEncoder(AudioEncoder):
    """Mean-pooling encoder for wav2vec2 / HuBERT / WavLM (self-supervised speech models).

    These produce per-timestep hidden states, so we masked-mean-pool over time to obtain a
    single clip embedding. Best suited for speech tasks (commands, speaker, emotion).
    """

    def __init__(self, model_id: str = "facebook/wav2vec2-base", device: Optional[str] = None) -> None:
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_id = model_id
        self.backbone = AutoModel.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.target_sr = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.embedding_dim = int(self.backbone.config.hidden_size)
        self.to(_resolve_device(device))

    def prepare(self, waveforms: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        return self.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

    def forward_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        attention_mask = inputs.get("attention_mask")
        outputs = self.backbone(**inputs)
        hidden = outputs.last_hidden_state  # [B, T, H]
        if attention_mask is None:
            return hidden.mean(dim=1)
        # Convert the sample-level attention mask to the conv-downsampled hidden resolution.
        out_lengths = self.backbone._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(hidden.device)
        time_idx = torch.arange(hidden.shape[1], device=hidden.device)
        mask = (time_idx[None, :] < out_lengths[:, None]).unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def save(self, save_directory: str) -> None:
        self.backbone.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)


class WhisperEncoder(AudioEncoder):
    """Whisper encoder (encoder tower only) as a fixed-size embedder.

    The log-mel features are always padded to 30s, so we mean-pool over the encoder's
    output frames. Robust to noisy speech.
    """

    def __init__(self, model_id: str = "openai/whisper-base", device: Optional[str] = None) -> None:
        super().__init__()
        from transformers import AutoFeatureExtractor, WhisperModel

        self.model_id = model_id
        # Keep the full WhisperModel so save/load round-trips cleanly; only the encoder is used.
        self.whisper = WhisperModel.from_pretrained(model_id)
        self.encoder = self.whisper.get_encoder()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.target_sr = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.embedding_dim = int(self.whisper.config.d_model)
        self.to(_resolve_device(device))

    def prepare(self, waveforms: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        return self.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )

    def forward_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(input_features=inputs["input_features"])
        return outputs.last_hidden_state.mean(dim=1)

    def save(self, save_directory: str) -> None:
        self.whisper.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)


# Registry mapping a transformers ``model_type`` to an encoder class.
_ENCODER_REGISTRY = {
    "clap": ClapAudioEncoder,
    "audio-spectrogram-transformer": ASTEncoder,
    "wav2vec2": Wav2Vec2LikeEncoder,
    "wav2vec2-conformer": Wav2Vec2LikeEncoder,
    "hubert": Wav2Vec2LikeEncoder,
    "wavlm": Wav2Vec2LikeEncoder,
    "unispeech": Wav2Vec2LikeEncoder,
    "unispeech-sat": Wav2Vec2LikeEncoder,
    "data2vec-audio": Wav2Vec2LikeEncoder,
    "whisper": WhisperEncoder,
}


def build_encoder(
    model_id: str, device: Optional[str] = None, encoder_type: Optional[str] = None, **kwargs
) -> AudioEncoder:
    """Build the appropriate `AudioEncoder` for a model id or local backbone directory.

    Selection order:
      1. explicit ``encoder_type`` (looked up in ``_ENCODER_REGISTRY``),
      2. the backbone's ``model_type`` via `AutoConfig`.

    To support a new architecture, implement a subclass and add it to ``_ENCODER_REGISTRY``.
    """
    if encoder_type is not None:
        if encoder_type not in _ENCODER_REGISTRY:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}. Known: {sorted(_ENCODER_REGISTRY)}.")
        return _ENCODER_REGISTRY[encoder_type](model_id, device=device, **kwargs)

    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id)
        model_type = getattr(config, "model_type", None)
    except Exception:
        model_type = None

    encoder_cls = _ENCODER_REGISTRY.get(model_type)
    if encoder_cls is None:
        supported = sorted(_ENCODER_REGISTRY)
        raise ValueError(
            f"No audiosetfit encoder registered for model_type={model_type!r} (model_id={model_id!r}). "
            f"Currently supported: {supported}. See the README for how to add new backbones."
        )
    return encoder_cls(model_id, device=device, **kwargs)
