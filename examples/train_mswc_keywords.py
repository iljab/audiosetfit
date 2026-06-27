"""Few-shot keyword spotting on MSWC with audiosetfit.

Multilingual Spoken Words Corpus (MSWC) is a keyword-spotting benchmark in the spirit of the
SUPERB KS task: each clip is a single spoken word, and the label is which word it is. This is a
*lexical/phonetic* task, so self-supervised speech encoders (wav2vec2 / HuBERT / WavLM) are a
strong default -- a nice contrast to the semantic sound-event tasks (ESC-50, UrbanSound8K).

The English config has 271 keywords with predefined train/validation/test splits; this script
restricts to a handful of keywords for a fast few-shot run.

Examples:
    python examples/train_mswc_keywords.py                       # 10 keywords, wav2vec2-base
    python examples/train_mswc_keywords.py --classes 5 --num-samples 16
    python examples/train_mswc_keywords.py --language spanish
    python examples/train_mswc_keywords.py --backbone laion/clap-htsat-unfused   # compare vs CLAP
    python examples/train_mswc_keywords.py --no-embedding-finetuning             # frozen baseline
"""

import argparse

from datasets import Audio, load_dataset

from audiosetfit import AudioSetFitModel, Trainer, TrainingArguments, sample_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot keyword spotting (MSWC) with audiosetfit")
    p.add_argument("--backbone", default="facebook/wav2vec2-base", help="HF audio backbone id")
    p.add_argument("--language", default="english", help="MSWC config: english / indian / spanish")
    p.add_argument("--classes", type=int, default=10, help="Number of keywords to use")
    p.add_argument("--num-samples", type=int, default=8, help="Labeled examples per class (few-shot)")
    p.add_argument("--epochs", type=int, default=1, help="Embedding fine-tuning epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Embedding (pair) batch size")
    p.add_argument("--max-steps", type=int, default=-1, help="Cap phase-1 optimizer steps (-1 = no cap)")
    p.add_argument("--eval-size", type=int, default=120, help="Max eval clips (for speed)")
    p.add_argument("--no-embedding-finetuning", action="store_true", help="Skip phase 1 (frozen backbone)")
    p.add_argument("--differentiable-head", action="store_true", help="Use a torch head instead of LogisticRegression")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (auto if omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs", type=int, default=256, help="Cap total contrastive pairs (-1 = no cap)")
    p.add_argument("--loss", default="cosine", help="Phase-1 loss: cosine / contrastive / supcon")
    p.add_argument("--samples-per-class", type=int, default=2, help="Examples per class per batch (supcon path)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for phase 1.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading MSWC keyword spotting (confit/mswc-parquet, config={args.language})...")
    train_pool = load_dataset("confit/mswc-parquet", args.language, split="train")
    test_set = load_dataset("confit/mswc-parquet", args.language, split="test")

    # The dataset ships an int `label` (ClassLabel) column; drop it so our column_mapping
    # ({"keyword": "label"}) doesn't collide with an existing `label` field.
    if "label" in train_pool.column_names:
        train_pool = train_pool.remove_columns("label")
    if "label" in test_set.column_names:
        test_set = test_set.remove_columns("label")

    # Pick a deterministic subset of keywords for a fast local run.
    selected = sorted(set(train_pool["keyword"]))[: args.classes]
    print(f"Using {len(selected)} keywords: {selected}")
    train_pool = train_pool.filter(lambda k: k in selected, input_columns="keyword")
    test_set = test_set.filter(lambda k: k in selected, input_columns="keyword")

    model = AudioSetFitModel.from_pretrained(
        args.backbone,
        labels=selected,
        use_differentiable_head=args.differentiable_head,
        device=args.device,
    )
    print(f"Backbone={args.backbone} | device={model.device} | embedding_dim={model.model_body.embedding_dim}")

    # Cast to the backbone's expected sample rate (wav2vec2/HuBERT/WavLM=16k, CLAP=48k).
    target_sr = model.model_body.target_sr
    train_pool = train_pool.cast_column("audio", Audio(sampling_rate=target_sr))
    test_set = test_set.cast_column("audio", Audio(sampling_rate=target_sr))

    train_ds = sample_dataset(train_pool, label_column="keyword", num_samples=args.num_samples, seed=args.seed)
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
        column_mapping={"keyword": "label"},  # 'audio' column already matches
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nEval metrics: {metrics}")

    sample = test_set.select(range(min(3, len(test_set))))
    preds = model.predict(list(sample["audio"]))
    print("\nSample predictions:")
    for true_label, pred in zip(sample["keyword"], preds):
        flag = "OK " if true_label == pred else "XX "
        print(f"  {flag} true={true_label:>15s}  pred={pred}")


if __name__ == "__main__":
    main()
