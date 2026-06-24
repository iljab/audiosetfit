"""Data utilities: few-shot sampling and audio loading/resampling.

Audio "examples" handled across the library can be any of:
  * a file path (`str` / `os.PathLike`),
  * a raw waveform (`np.ndarray`, assumed already at the encoder's target sample rate),
  * a Hugging Face `datasets` Audio dict: ``{"array": np.ndarray, "sampling_rate": int, "path": str}``.

`load_audio` normalizes all of these into a mono float32 waveform at a target sample rate.
"""

from __future__ import annotations

import os
from typing import Any, List, Union

import numpy as np

AudioInput = Union[str, os.PathLike, np.ndarray, dict]


def _to_mono(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        # (channels, samples) or (samples, channels) -> average to mono
        axis = 0 if array.shape[0] < array.shape[1] else 1
        array = array.mean(axis=axis)
    return array.astype(np.float32, copy=False)


def load_audio(example: AudioInput, target_sr: int) -> np.ndarray:
    """Return a mono float32 waveform at ``target_sr`` from any supported input type."""
    # torchcodec AudioDecoder (datasets >= 4 lazy audio decoding)
    if hasattr(example, "get_all_samples"):
        samples = example.get_all_samples()
        array = _to_mono(np.asarray(samples.data.detach().cpu()))
        sr = int(samples.sample_rate)
        if sr != target_sr:
            array = _resample(array, sr, target_sr)
        return array

    # Hugging Face Audio feature dict
    if isinstance(example, dict):
        array = _to_mono(example["array"])
        sr = int(example.get("sampling_rate", target_sr))
        if sr != target_sr:
            array = _resample(array, sr, target_sr)
        return array

    # Raw waveform: assumed to already be at target_sr
    if isinstance(example, np.ndarray):
        return _to_mono(example)

    # File path
    if isinstance(example, (str, os.PathLike)):
        import librosa

        array, _ = librosa.load(os.fspath(example), sr=target_sr, mono=True)
        return array.astype(np.float32, copy=False)

    raise TypeError(
        f"Unsupported audio input type: {type(example)!r}. "
        "Expected a file path, a numpy waveform, or a datasets Audio dict."
    )


def _resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import librosa

    return librosa.resample(array, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32, copy=False)


def load_audio_batch(examples: List[AudioInput], target_sr: int) -> List[np.ndarray]:
    return [load_audio(ex, target_sr) for ex in examples]


def sample_dataset(dataset, label_column: str = "label", num_samples: int = 8, seed: int = 42):
    """Sample (at most) ``num_samples`` examples per class to simulate the few-shot regime.

    Mirrors `setfit.sample_dataset`. Requires the `datasets` library.
    """
    from datasets import Dataset

    shuffled = dataset.shuffle(seed=seed)
    df = shuffled.to_pandas()
    df = df.groupby(label_column).head(n=num_samples).reset_index(drop=True)
    sampled = Dataset.from_pandas(df, features=dataset.features)
    return sampled.shuffle(seed=seed)


def infer_labels(values: List[Any]) -> List[str]:
    """Return the sorted unique string labels present in ``values`` (best-effort)."""
    uniques = sorted(set(values), key=lambda x: (isinstance(x, str), x))
    return [str(u) for u in uniques]
