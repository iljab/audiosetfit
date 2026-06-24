# AudioSetFit

**Efficient few-shot audio classification with contrastive fine-tuning** - a [SetFit](https://github.com/huggingface/setfit) for audio.

`audiosetfit` ports SetFit's prompt-free, few-shot recipe from text to audio. Instead of a
`SentenceTransformer` body, it uses an **audio encoder** (CLAP by default) and trains in two
phases:

1. **Embedding fine-tuning (contrastive).** From a handful of labeled clips it builds
  *positive* (same-class) and *negative* (different-class) pairs and fine-tunes the audio
   body so same-class clips embed closer together. A few examples explode into hundreds of
   informative pairs.
2. **Classifier head.** A lightweight head (sklearn `LogisticRegression` by default, or a
  differentiable torch head) is fit on the resulting embeddings.

The contrastive trainer is **self-contained**, it does *not* depend on
`sentence-transformers`. The pair-sampling and loss math are reimplemented to operate
directly on audio embeddings, so any HF audio model can be plugged in as the body.

The public API intentionally mirrors SetFit:

```python
from audiosetfit import AudioSetFitModel, Trainer, TrainingArguments, sample_dataset
```

## Installation

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

This installs `torch`, `transformers`, `datasets`, `librosa`, `soundfile`, `torchcodec`,
`scikit-learn`, etc. On Apple Silicon, PyTorch will use the **MPS** backend automatically;
on NVIDIA GPUs it uses CUDA; otherwise CPU.

> **FFmpeg required.** `datasets >= 4` decodes audio via `torchcodec`, which needs FFmpeg
> (4–7) installed on your system. On macOS: `brew install ffmpeg`; on Debian/Ubuntu:
> `sudo apt-get install ffmpeg`.

## Quickstart: few-shot ESC-50

ESC-50 (2,000 environmental-sound clips, 50 classes) is the small default starting point. The example restricts to a few classes for a fast first run:

```bash
python examples/train_esc50.py                      # 5 classes, 8 shots, CLAP
python examples/train_esc50.py --classes 10 --num-samples 16
python examples/train_esc50.py --no-embedding-finetuning   # frozen-backbone baseline
```

Minimal end-to-end usage:

```python
from datasets import Audio, load_dataset
from audiosetfit import AudioSetFitModel, Trainer, TrainingArguments, sample_dataset

ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(sampling_rate=48000))
labels = sorted(set(ds["category"]))
train_ds = sample_dataset(ds, label_column="category", num_samples=8)

model = AudioSetFitModel.from_pretrained("laion/clap-htsat-unfused", labels=labels)
trainer = Trainer(
    model=model,
    args=TrainingArguments(embedding_num_epochs=1, max_steps=60),
    train_dataset=train_ds,
    column_mapping={"category": "label"},  # the 'audio' column already matches
)
trainer.train()

preds = model.predict(["dog_bark.wav", "rain.wav"])     # file paths, arrays, or Audio dicts
model.save_pretrained("my-esc50-model")
reloaded = AudioSetFitModel.from_pretrained("my-esc50-model")
```

### Inputs accepted everywhere

`predict` / `encode` / datasets accept any mix of:

- file paths (`"clip.wav"`),
- raw waveforms (`np.ndarray`, assumed at the backbone's sample rate),
- Hugging Face `datasets` Audio dicts (`{"array", "sampling_rate", "path"}`).

Everything is resampled to the backbone's expected rate (CLAP = 48 kHz).

## Why CLAP as the default backbone?

[CLAP](https://huggingface.co/laion/clap-htsat-unfused) (Contrastive Language-Audio
Pretraining) is the closest audio analog to a sentence-transformer: it was *already* trained
contrastively, producing a 512-d projection space where semantically similar sounds are close
together. That makes it an excellent starting point for few-shot contrastive fine-tuning and
even the frozen-backbone baseline (`--no-embedding-finetuning`) is strong.

## Project layout

```
src/audiosetfit/
├── encoders.py      # AudioEncoder base + CLAP/AST/wav2vec2-family/Whisper + build_encoder()
├── modeling.py      # AudioSetFitModel, AudioSetFitHead, save/from_pretrained
├── sampler.py       # ContrastiveDataset (same/different-label pair generation)
├── losses.py        # CosineSimilarityLoss, ContrastiveLoss (on embedding tensors)
├── data.py          # load_audio (resampling), sample_dataset
├── training_args.py # TrainingArguments (both phases)
└── trainer.py       # self-contained two-phase Trainer
examples/train_esc50.py
```

## Key training arguments


| Argument                  | Default          | Purpose                                                  |
| ------------------------- | ---------------- | -------------------------------------------------------- |
| `train_embeddings`        | `True`           | Run phase 1. Set `False` for a frozen-backbone baseline. |
| `embedding_num_epochs`    | `1`              | Epochs over contrastive pairs.                           |
| `embedding_batch_size`    | `16`             | Pair batch size (lower it if you hit memory limits).     |
| `body_learning_rate`      | `2e-5`           | LR for the audio body.                                   |
| `loss`                    | `"cosine"`       | `"cosine"` or `"contrastive"` (or pass an `nn.Module`).  |
| `sampling_strategy`       | `"oversampling"` | `"unique"` / `"oversampling"` / `"undersampling"`.       |
| `max_steps` / `max_pairs` | `-1`             | Cap phase-1 work (handy on CPU/laptops).                 |
| `classifier_num_epochs`   | `25`             | Torch-head epochs (ignored for sklearn head).            |


## Backbones

Pick a backbone by passing its Hugging Face id to `from_pretrained` (or `--backbone` in the
example). The right `AudioEncoder` is selected automatically from the model's `model_type`.

```python
AudioSetFitModel.from_pretrained("laion/clap-htsat-unfused")                  # CLAP (default)
AudioSetFitModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")   # AST
AudioSetFitModel.from_pretrained("facebook/wav2vec2-base")                    # wav2vec2
AudioSetFitModel.from_pretrained("facebook/hubert-base-ls960")                # HuBERT
AudioSetFitModel.from_pretrained("microsoft/wavlm-base-plus")                 # WavLM
AudioSetFitModel.from_pretrained("openai/whisper-base")                       # Whisper encoder
```

```bash
python examples/train_esc50.py --backbone facebook/wav2vec2-base
python examples/train_esc50.py --backbone MIT/ast-finetuned-audioset-10-10-0.4593
```

Audio is resampled to each backbone's expected rate automatically (CLAP 48 kHz, others
16 kHz), so the same dataset works across all of them.


| Backbone                      | `model_type` (built-in)                                                                                  | Embedding         | Pooling               | Best for                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------- | --------------------- | ----------------------------------------- |
| **CLAP** (default)            | `clap`                                                                                                   | 512-d, normalized | projection head       | General sound events, environmental audio |
| **AST**                       | `audio-spectrogram-transformer`                                                                          | 768-d             | CLS+dist pooled       | AudioSet-style tagging                    |
| **wav2vec2 / HuBERT / WavLM** | `wav2vec2` / `hubert` / `wavlm` (+ `unispeech`, `unispeech-sat`, `data2vec-audio`, `wav2vec2-conformer`) | hidden_size       | masked mean over time | Speech (commands, speaker, emotion)       |
| **Whisper encoder**           | `whisper`                                                                                                | d_model           | mean over frames      | Robust speech in noise                    |


### Adding another backbone

The encoder is the only modality-specific piece. Subclass `AudioEncoder`, implement
`prepare` (waveforms → model inputs) and `forward_features` (inputs → `[B, D]`), set
`target_sr` / `embedding_dim`, then register it:

```python
from audiosetfit import encoders

class MyEncoder(encoders.AudioEncoder):
    def __init__(self, model_id, device=None):
        super().__init__()
        self.model_id = model_id
        ...                       # load backbone + feature extractor
        self.target_sr = 16000
        self.embedding_dim = ...  # output dim
        self.to(encoders._resolve_device(device))
    def prepare(self, waveforms): ...
    def forward_features(self, inputs): ...
    def save(self, save_directory): ...

encoders._ENCODER_REGISTRY["my_model_type"] = MyEncoder
```

## Roadmap / next steps

- (Optional) BEATs / OpenBEATs backbone: strongest general-purpose SSL audio embeddings;
needs external modeling code + a manually downloaded checkpoint, so deferred for now.
- `SupConLoss` + group-by-label batch sampler (as in SetFit).
- Multilabel audio tagging end-to-end example.
- Other dataset examples: e.g., Speech Commands example.
- ONNX export and Hub `push_to_hub`.

## Acknowledgements

Architecture and training recipe adapted from
[Hugging Face SetFit](https://github.com/huggingface/setfit)
(Tunstall et al., *Efficient Few-Shot Learning Without Prompts*, 2022).