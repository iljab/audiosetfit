"""Self-contained two-phase trainer (no sentence-transformers dependency).

Phase 1 — embedding fine-tuning: builds contrastive (same/different label) pairs and
fine-tunes the audio body so same-class clips embed closer together.
Phase 2 — classifier head: fits the head on embeddings of the training examples.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from .data import load_audio_batch
from .losses import get_loss
from .modeling import AudioSetFitModel
from .sampler import ContrastiveDataset
from .training_args import TrainingArguments


class _PairDataset(Dataset):
    """Map-style dataset over contrastive index pairs, returning (wave_a, wave_b, label)."""

    def __init__(self, pairs: List[Dict], waveforms: List[np.ndarray]) -> None:
        self.pairs = pairs
        self.waveforms = waveforms

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        return self.waveforms[pair["idx_1"]], self.waveforms[pair["idx_2"]], float(pair["label"])


class Trainer:
    """Trainer to fit an `AudioSetFitModel` from few labeled audio examples."""

    _REQUIRED_COLUMNS = {"audio", "label"}

    def __init__(
        self,
        model: AudioSetFitModel,
        args: Optional[TrainingArguments] = None,
        train_dataset=None,
        eval_dataset=None,
        metric: Union[str, Callable] = "accuracy",
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.args = args or TrainingArguments()
        self.metric = metric
        self.column_mapping = column_mapping
        self.train_dataset = self._apply_column_mapping(train_dataset) if train_dataset is not None else None
        self.eval_dataset = self._apply_column_mapping(eval_dataset) if eval_dataset is not None else None

    # ------------------------------------------------------------------ columns
    def _apply_column_mapping(self, dataset):
        if self.column_mapping:
            rename = {src: dst for src, dst in self.column_mapping.items() if src in dataset.column_names}
            dataset = dataset.rename_columns(rename)
        missing = self._REQUIRED_COLUMNS - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"The dataset is missing required columns {sorted(missing)}. "
                f"Found {sorted(dataset.column_names)}. Provide a `column_mapping` mapping your "
                "audio/label columns to 'audio' and 'label'."
            )
        return dataset

    def _dataset_to_xy(self, dataset):
        return list(dataset["audio"]), list(dataset["label"])

    # ------------------------------------------------------------------ labels
    def _ensure_labels(self, y: List) -> None:
        if self.model.multi_target_strategy is not None:
            return
        if self.model.labels is None:
            uniques = sorted({str(v) for v in y})
            self.model.labels = uniques

    def _encode_labels(self, y: List) -> List[int]:
        if self.model.multi_target_strategy is not None:
            return y
        label2id = self.model.label2id
        return [label2id[str(v)] for v in y]

    # ------------------------------------------------------------------ training
    def train(self, args: Optional[TrainingArguments] = None) -> None:
        args = args or self.args
        if self.train_dataset is None:
            raise ValueError("Training requires a `train_dataset`.")
        x_train, y_train = self._dataset_to_xy(self.train_dataset)
        self._ensure_labels(y_train)
        y_enc = self._encode_labels(y_train)

        if args.train_embeddings:
            self.train_embeddings(x_train, y_enc, args)
        self.train_classifier(x_train, y_enc, args)

    def train_embeddings(self, x_train: List, y_train: List, args: TrainingArguments) -> None:
        body = self.model.model_body
        self.model.unfreeze("body")
        body.train()

        max_pairs = args.max_pairs
        if max_pairs == -1 and args.max_steps != -1:
            max_pairs = args.max_steps * args.embedding_batch_size

        contrastive = ContrastiveDataset(
            labels=y_train,
            multilabel=self.model.multi_target_strategy is not None,
            num_iterations=args.num_iterations,
            sampling_strategy=args.sampling_strategy,
            max_pairs=max_pairs,
        )
        pairs = list(contrastive)
        waveforms = load_audio_batch(x_train, body.target_sr)
        dataset = _PairDataset(pairs, waveforms)
        dataloader = DataLoader(
            dataset,
            batch_size=args.embedding_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=self._collate_pairs,
        )

        loss_fn = get_loss(args.loss)
        if hasattr(loss_fn, "margin"):
            try:
                loss_fn.margin = args.margin
            except Exception:
                pass
        loss_fn.to(self.model.device)

        params = [p for p in body.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.body_learning_rate, weight_decay=args.l2_weight)

        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * args.embedding_num_epochs
        if args.max_steps != -1:
            total_steps = min(total_steps, args.max_steps)
        scheduler = self._build_scheduler(optimizer, total_steps, args.warmup_proportion)

        use_amp = args.use_amp and self.model.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        print(f"***** Embedding fine-tuning *****")
        print(f"  Num pairs        = {len(pairs)}")
        print(f"  Batch size       = {args.embedding_batch_size}")
        print(f"  Epochs           = {args.embedding_num_epochs}")
        print(f"  Total optim steps= {total_steps}")

        global_step = 0
        for _ in trange(args.embedding_num_epochs, desc="Embedding epoch", disable=not args.show_progress_bar):
            for batch in tqdm(dataloader, desc="Pairs", leave=False, disable=not args.show_progress_bar):
                if args.max_steps != -1 and global_step >= args.max_steps:
                    break
                inputs_a, inputs_b, labels = batch
                inputs_a = {k: v.to(self.model.device) for k, v in inputs_a.items()}
                inputs_b = {k: v.to(self.model.device) for k, v in inputs_b.items()}
                labels = labels.to(self.model.device)

                optimizer.zero_grad()
                with torch.autocast(device_type=self.model.device.type, enabled=use_amp):
                    emb_a = body.forward_features(inputs_a)
                    emb_b = body.forward_features(inputs_b)
                    loss = loss_fn(emb_a, emb_b, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
            if args.max_steps != -1 and global_step >= args.max_steps:
                break

        body.eval()

    def _collate_pairs(self, batch):
        waves_a, waves_b, labels = zip(*batch)
        inputs_a = self.model.model_body.prepare(list(waves_a))
        inputs_b = self.model.model_body.prepare(list(waves_b))
        labels = torch.tensor(labels, dtype=torch.float32)
        return inputs_a, inputs_b, labels

    def _build_scheduler(self, optimizer, total_steps: int, warmup_proportion: float):
        warmup_steps = int(max(total_steps, 1) * warmup_proportion)
        try:
            from transformers import get_linear_schedule_with_warmup

            return get_linear_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))
        except Exception:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def train_classifier(self, x_train: List, y_train: List, args: TrainingArguments) -> None:
        self.model.fit(
            x_train,
            y_train,
            num_epochs=args.classifier_num_epochs,
            batch_size=args.classifier_batch_size,
            head_learning_rate=args.head_learning_rate,
            l2_weight=args.l2_weight,
            show_progress_bar=args.show_progress_bar,
        )

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, dataset=None, metric_key_prefix: str = "test") -> Dict[str, float]:
        dataset = self._apply_column_mapping(dataset) if dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided.")
        x_test, y_test = self._dataset_to_xy(dataset)
        y_true = self._encode_labels(y_test)

        y_pred = self.model.predict(x_test, use_labels=False, show_progress_bar=self.args.show_progress_bar)
        y_pred = np.asarray(y_pred)

        if callable(self.metric):
            results = self.metric(y_pred, np.asarray(y_true))
            return results if isinstance(results, dict) else {f"{metric_key_prefix}_metric": results}

        from sklearn.metrics import accuracy_score, f1_score

        if self.metric == "accuracy":
            return {f"{metric_key_prefix}_accuracy": float(accuracy_score(y_true, y_pred))}
        if self.metric == "f1":
            return {f"{metric_key_prefix}_f1": float(f1_score(y_true, y_pred, average="macro"))}
        raise ValueError(f"Unknown metric {self.metric!r}. Use 'accuracy', 'f1', or a callable.")

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        raise NotImplementedError(
            "Hub upload is not implemented in 0.1.0. Use `model.save_pretrained(dir)` and upload manually."
        )
