"""Few-shot speech emotion recognition on CREMA-D with audiosetfit.

CREMA-D (7,442 short clips, 91 actors, 6 emotions: anger/disgust/fear/happy/neutral/sad) is a
paralinguistic task: the *content* of the speech is fixed, and the label depends on *how* it is
spoken. This is exactly where self-supervised speech encoders (wav2vec2 / HuBERT / WavLM) tend
to beat a general audio-text model like CLAP, so this example defaults to ``facebook/wav2vec2-base``.

To keep evaluation honest, the train/test split is *speaker-disjoint*: a fraction of the actors
is held out entirely for testing, so the model is scored on voices it never heard during the
few-shot fit. The speaker id is the first token of each filename (e.g. ``1068_TIE_ANG_XX.wav``).

Backbone choice dominates here: a generic SSL model (wav2vec2/HuBERT/WavLM-base) only reaches
~0.3 accuracy at 8 shots, while a backbone already *task-pretrained* for emotion roughly doubles
that (e.g. ``Hatman/audio-emotion-detection``, a wav2vec2-xlsr-53 fine-tuned on Common Voice -- not
CREMA-D, so it is a fair cross-corpus transfer). Contrastive fine-tuning then adds a smaller bump.

Examples:
    python examples/train_cremad.py                              # wav2vec2-base, 8 shots/emotion
    python examples/train_cremad.py --backbone Hatman/audio-emotion-detection   # emotion-pretrained (best)
    python examples/train_cremad.py --backbone microsoft/wavlm-base
    python examples/train_cremad.py --backbone facebook/hubert-base-ls960
    python examples/train_cremad.py --backbone laion/clap-htsat-unfused   # compare vs CLAP
    python examples/train_cremad.py --no-embedding-finetuning             # frozen-backbone baseline
"""

import argparse
import os

from datasets import Audio, load_dataset

from audiosetfit import AudioSetFitModel, Trainer, TrainingArguments, sample_dataset


def speaker_id(path: str) -> str:
    """CREMA-D filenames look like ``1068_TIE_ANG_XX.wav``; the leading token is the actor id."""
    return os.path.basename(path).split("_")[0]


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot speech emotion recognition (CREMA-D) with audiosetfit")
    p.add_argument("--backbone", default="facebook/wav2vec2-base", help="HF audio backbone id")
    p.add_argument("--classes", type=int, default=6, help="Number of emotions to use (<=6)")
    p.add_argument("--num-samples", type=int, default=8, help="Labeled examples per class (few-shot)")
    p.add_argument("--epochs", type=int, default=1, help="Embedding fine-tuning epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Embedding (pair/group) batch size")
    p.add_argument("--max-steps", type=int, default=-1, help="Cap phase-1 optimizer steps (-1 = no cap)")
    p.add_argument("--eval-size", type=int, default=80, help="Max eval clips (for speed)")
    p.add_argument(
        "--test-speaker-frac",
        type=float,
        default=0.2,
        help="Fraction of actors held out (speaker-disjoint) for the test split.",
    )
    p.add_argument("--no-embedding-finetuning", action="store_true", help="Skip phase 1 (frozen backbone)")
    p.add_argument("--differentiable-head", action="store_true", help="Use a torch head instead of LogisticRegression")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (auto if omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs", type=int, default=256, help="Cap total contrastive pairs (-1 = no cap)")
    p.add_argument("--loss", default="supcon", help="Phase-1 loss: cosine / contrastive / supcon")
    p.add_argument("--samples-per-class", type=int, default=2, help="Examples per class per batch (supcon path)")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for phase 1.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading CREMA-D (confit/cremad-parquet)...")
    ds = load_dataset("confit/cremad-parquet", split="train")

    # The dataset ships an int `label` (ClassLabel) column; drop it so our column_mapping
    # ({"emotion": "label"}) doesn't collide with an existing `label` field.
    if "label" in ds.column_names:
        ds = ds.remove_columns("label")

    # Pick a deterministic subset of emotions for a fast local run.
    all_emotions = sorted(set(ds["emotion"]))
    selected = all_emotions[: args.classes]
    print(f"Using {len(selected)} emotions: {selected}")
    ds = ds.filter(lambda e: e in selected, input_columns="emotion")

    model = AudioSetFitModel.from_pretrained(
        args.backbone,
        labels=selected,
        use_differentiable_head=args.differentiable_head,
        device=args.device,
    )
    print(f"Backbone={args.backbone} | device={model.device} | embedding_dim={model.model_body.embedding_dim}")

    # Cast to the backbone's expected sample rate (wav2vec2/HuBERT/WavLM=16k, CLAP=48k).
    target_sr = model.model_body.target_sr
    ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))

    # Speaker-disjoint split: hold out a fraction of actors entirely for testing.
    actors = sorted({speaker_id(f) for f in ds["file"]})
    n_test = max(1, round(len(actors) * args.test_speaker_frac))
    test_actors = set(actors[-n_test:])
    print(f"{len(actors)} actors | {len(test_actors)} held out for the (speaker-independent) test split")

    train_pool = ds.filter(lambda f: speaker_id(f) not in test_actors, input_columns="file")
    test_set = ds.filter(lambda f: speaker_id(f) in test_actors, input_columns="file")

    train_ds = sample_dataset(train_pool, label_column="emotion", num_samples=args.num_samples, seed=args.seed)
    if args.eval_size > 0 and len(test_set) > args.eval_size:
        test_set = test_set.shuffle(seed=args.seed).select(range(args.eval_size))

    print(f"Train examples: {len(train_ds)} | Eval examples: {len(test_set)}")

    training_args = TrainingArguments(
        train_embeddings=not args.no_embedding_finetuning,
        embedding_num_epochs=args.epochs,
        embedding_batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        sampling_strategy="oversampling",
        loss=args.loss,
        samples_per_class=args.samples_per_class,
        num_workers=args.num_workers,
        max_pairs=args.max_pairs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_set,
        metric="accuracy",
        column_mapping={"emotion": "label"},  # 'audio' column already matches
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nEval metrics: {metrics}")

    # Inference on a few held-out clips.
    sample = test_set.select(range(min(3, len(test_set))))
    preds = model.predict(list(sample["audio"]))
    print("\nSample predictions:")
    for true_label, pred in zip(sample["emotion"], preds):
        flag = "OK " if true_label == pred else "XX "
        print(f"  {flag} true={true_label:>10s}  pred={pred}")


if __name__ == "__main__":
    main()
