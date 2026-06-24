"""Few-shot audio classification on ESC-50 with audiosetfit.

ESC-50 is the smaller of the two datasets we considered (2,000 clips, 50 classes, 5s each),
which makes it ideal for a first local run. By default this script restricts training to a
handful of classes so it finishes quickly on a laptop (CPU / Apple MPS); scale it up with
the CLI flags below.

Examples:
    python examples/train_esc50.py                       # 5 classes, 8 shots, CLAP
    python examples/train_esc50.py --classes 10 --num-samples 16
    python examples/train_esc50.py --no-embedding-finetuning   # frozen-backbone baseline
    python examples/train_esc50.py --max-steps 50              # cap phase-1 steps
"""

import argparse

from datasets import Audio, load_dataset

from audiosetfit import AudioSetFitModel, Trainer, TrainingArguments, sample_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot ESC-50 with audiosetfit")
    p.add_argument("--backbone", default="laion/clap-htsat-unfused", help="HF audio backbone id")
    p.add_argument("--classes", type=int, default=5, help="Number of ESC-50 classes to use (<=50)")
    p.add_argument("--num-samples", type=int, default=8, help="Labeled examples per class (few-shot)")
    p.add_argument("--epochs", type=int, default=1, help="Embedding fine-tuning epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Embedding (pair) batch size")
    p.add_argument("--max-steps", type=int, default=60, help="Cap phase-1 optimizer steps (-1 = no cap)")
    p.add_argument("--eval-size", type=int, default=80, help="Max eval clips (for speed)")
    p.add_argument("--no-embedding-finetuning", action="store_true", help="Skip phase 1 (frozen backbone)")
    p.add_argument("--differentiable-head", action="store_true", help="Use a torch head instead of LogisticRegression")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (auto if omitted)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading ESC-50 (ashraq/esc50)...")
    ds = load_dataset("ashraq/esc50", split="train")

    # Pick a deterministic subset of classes for a fast local run.
    all_categories = sorted(set(ds["category"]))
    selected = all_categories[: args.classes]
    print(f"Using {len(selected)} classes: {selected}")
    ds = ds.filter(lambda c: c in selected, input_columns="category")

    model = AudioSetFitModel.from_pretrained(
        #args.backbone,
        "/Users/iljabaumann/Downloads/BEATs_iter3.pt",
        encoder_type="beats",
        labels=selected,
        use_differentiable_head=args.differentiable_head,
        device=args.device,
    )
    print(f"Backbone={args.backbone} | device={model.device} | embedding_dim={model.model_body.embedding_dim}")

    # Cast to the backbone's expected sample rate (CLAP=48k, AST/wav2vec2/Whisper=16k).
    target_sr = model.model_body.target_sr
    ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))

    # Standard ESC-50 protocol: folds 1-4 for training pool, fold 5 for testing.
    train_pool = ds.filter(lambda f: f != 5, input_columns="fold")
    test_set = ds.filter(lambda f: f == 5, input_columns="fold")

    train_ds = sample_dataset(train_pool, label_column="category", num_samples=args.num_samples, seed=args.seed)
    if args.eval_size > 0 and len(test_set) > args.eval_size:
        test_set = test_set.shuffle(seed=args.seed).select(range(args.eval_size))

    print(f"Train examples: {len(train_ds)} | Eval examples: {len(test_set)}")

    training_args = TrainingArguments(
        train_embeddings=not args.no_embedding_finetuning,
        embedding_num_epochs=args.epochs,
        embedding_batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        sampling_strategy="unique",
        loss="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_set,
        metric="accuracy",
        column_mapping={"category": "label"},  # 'audio' column already matches
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nEval metrics: {metrics}")

    # Inference on a few held-out clips.
    sample = test_set.select(range(min(3, len(test_set))))
    preds = model.predict(list(sample["audio"]))
    print("\nSample predictions:")
    for true_label, pred in zip(sample["category"], preds):
        flag = "OK " if true_label == pred else "XX "
        print(f"  {flag} true={true_label:>20s}  pred={pred}")


if __name__ == "__main__":
    main()
