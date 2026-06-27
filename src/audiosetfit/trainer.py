"""Self-contained two-phase trainer (no sentence-transformers dependency).

Phase 1 — embedding fine-tuning: builds contrastive (same/different label) pairs and
fine-tunes the audio body so same-class clips embed closer together.
Phase 2 — classifier head: fits the head on embeddings of the training examples.
"""

from __future__ import annotations

import random
from collections import defaultdict
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


class _PairCollator:
    """Top-level (picklable) collate fn that runs the encoder's feature extraction on a batch.

    It holds only the audio encoder *body* — not the `Trainer`, its datasets, or the head — so
    that with ``num_workers > 0`` the `DataLoader` doesn't try to pickle the whole training
    state. Note: under the ``spawn`` start method (macOS/Windows) the body is still serialized
    to each worker, so prefer ``num_workers=0`` for large backbones there; on Linux (``fork``)
    workers share memory and ``num_workers > 0`` is cheap.
    """

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def __call__(self, batch):
        waves_a, waves_b, labels = zip(*batch)
        inputs_a = self.encoder.prepare(list(waves_a))
        inputs_b = self.encoder.prepare(list(waves_b))
        labels = torch.tensor(labels, dtype=torch.float32)
        return inputs_a, inputs_b, labels


class _SampleDataset(Dataset):
    """Map-style dataset over single (waveform, int-label) examples for the in-batch loss path."""

    def __init__(self, waveforms: List[np.ndarray], labels: List[int]) -> None:
        self.waveforms = waveforms
        self.labels = labels

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int):
        return self.waveforms[idx], int(self.labels[idx])


class _SampleCollator:
    """Top-level (picklable) collate fn that feature-extracts a batch of single examples."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def __call__(self, batch):
        waves, labels = zip(*batch)
        inputs = self.encoder.prepare(list(waves))
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs, labels


class _GroupByLabelBatchSampler:
    """Yields index batches of ``num_classes_per_batch`` classes x ``samples_per_class`` examples.

    Guarantees each batch contains in-batch positives (>=2 per class) and negatives (>=2 classes),
    which is what makes a supervised-contrastive loss meaningful. Classes with too few examples
    are sampled with replacement so the few-shot regime still works.
    """

    def __init__(
        self,
        labels: List[int],
        num_classes_per_batch: int,
        samples_per_class: int,
        num_batches: int,
        seed: int = 42,
    ) -> None:
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, y in enumerate(labels):
            self.label_to_indices[int(y)].append(i)
        self.classes = list(self.label_to_indices)
        self.num_classes_per_batch = max(2, min(num_classes_per_batch, len(self.classes)))
        self.samples_per_class = max(2, samples_per_class)
        self.num_batches = max(1, num_batches)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.sample(self.classes, self.num_classes_per_batch)
            batch: List[int] = []
            for c in chosen:
                idxs = self.label_to_indices[c]
                if len(idxs) >= self.samples_per_class:
                    batch.extend(self.rng.sample(idxs, self.samples_per_class))
                else:
                    batch.extend(self.rng.choice(idxs) for _ in range(self.samples_per_class))
            yield batch


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

        loss_fn = get_loss(args.loss)
        if hasattr(loss_fn, "margin"):
            try:
                loss_fn.margin = args.margin
            except Exception:
                pass
        if hasattr(loss_fn, "temperature"):
            try:
                loss_fn.temperature = args.supcon_temperature
            except Exception:
                pass
        loss_fn.to(self.model.device)

        waveforms = load_audio_batch(x_train, body.target_sr)
        in_batch = getattr(loss_fn, "in_batch", False)
        if in_batch:
            dataloader, data_desc = self._build_supcon_loader(waveforms, y_train, args)
        else:
            dataloader, data_desc = self._build_pair_loader(waveforms, y_train, args)

        params = [p for p in body.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.body_learning_rate, weight_decay=args.l2_weight)

        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * args.embedding_num_epochs
        if args.max_steps != -1:
            total_steps = min(total_steps, args.max_steps)
        scheduler = self._build_scheduler(optimizer, total_steps, args.warmup_proportion)

        use_amp = args.use_amp and self.model.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        print("***** Embedding fine-tuning *****")
        print(f"  Loss             = {type(loss_fn).__name__}")
        print(f"  {data_desc}")
        print(f"  Batch size       = {args.embedding_batch_size}")
        print(f"  Epochs           = {args.embedding_num_epochs}")
        print(f"  Total optim steps= {total_steps}")

        global_step = 0
        for _ in trange(args.embedding_num_epochs, desc="Embedding epoch", disable=not args.show_progress_bar):
            for batch in tqdm(dataloader, desc="Steps", leave=False, disable=not args.show_progress_bar):
                if args.max_steps != -1 and global_step >= args.max_steps:
                    break
                optimizer.zero_grad()
                with torch.autocast(device_type=self.model.device.type, enabled=use_amp):
                    if in_batch:
                        inputs, labels = batch
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        labels = labels.to(self.model.device)
                        emb = body.forward_features(inputs)
                        loss = loss_fn(emb, labels)
                    else:
                        inputs_a, inputs_b, labels = batch
                        inputs_a = {k: v.to(self.model.device) for k, v in inputs_a.items()}
                        inputs_b = {k: v.to(self.model.device) for k, v in inputs_b.items()}
                        labels = labels.to(self.model.device)
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

    def _build_pair_loader(self, waveforms: List[np.ndarray], y_train: List[int], args: TrainingArguments):
        """Pairwise path: build same/different-label pairs (cosine / contrastive losses)."""
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
        dataset = _PairDataset(pairs, waveforms)
        dataloader = DataLoader(
            dataset,
            batch_size=args.embedding_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=_PairCollator(self.model.model_body),
        )
        return dataloader, f"Num pairs        = {len(pairs)}"

    def _build_supcon_loader(self, waveforms: List[np.ndarray], y_train: List[int], args: TrainingArguments):
        """In-batch path: group-by-label batches so every batch has positives and negatives."""
        samples_per_class = max(2, args.samples_per_class)
        n_classes = len(set(int(y) for y in y_train))
        num_classes_per_batch = max(2, min(args.embedding_batch_size // samples_per_class, n_classes))
        effective_batch = num_classes_per_batch * samples_per_class

        # Mirror the pairwise path's budget: reuse max_steps / max_pairs to size steps-per-epoch.
        if args.max_steps != -1:
            steps_per_epoch = args.max_steps
        elif args.max_pairs != -1:
            steps_per_epoch = max(1, args.max_pairs // effective_batch)
        else:
            steps_per_epoch = max(1, len(waveforms) // effective_batch)

        sampler = _GroupByLabelBatchSampler(
            labels=y_train,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
            num_batches=steps_per_epoch,
            seed=args.seed,
        )
        dataset = _SampleDataset(waveforms, y_train)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=_SampleCollator(self.model.model_body),
        )
        desc = f"Batches/epoch    = {steps_per_epoch} ({num_classes_per_batch} classes x {samples_per_class})"
        return dataloader, desc

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
    def _predict_true(self, dataset=None):
        """Return ``(y_true, y_pred)`` as encoded-int numpy arrays over an eval dataset."""
        dataset = self._apply_column_mapping(dataset) if dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("No evaluation dataset provided.")
        x_test, y_test = self._dataset_to_xy(dataset)
        y_true = np.asarray(self._encode_labels(y_test))
        y_pred = np.asarray(
            self.model.predict(x_test, use_labels=False, show_progress_bar=self.args.show_progress_bar)
        )
        return y_true, y_pred

    @torch.no_grad()
    def evaluate(self, dataset=None, metric_key_prefix: str = "test") -> Dict[str, float]:
        y_true, y_pred = self._predict_true(dataset)

        if callable(self.metric):
            results = self.metric(y_pred, y_true)
            return results if isinstance(results, dict) else {f"{metric_key_prefix}_metric": results}

        from sklearn.metrics import accuracy_score, f1_score

        if self.metric in ("accuracy", "f1"):
            # Report both so a single run is informative regardless of the requested metric.
            return {
                f"{metric_key_prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
                f"{metric_key_prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            }
        raise ValueError(f"Unknown metric {self.metric!r}. Use 'accuracy', 'f1', or a callable.")

    @torch.no_grad()
    def classification_report(self, dataset=None) -> Dict[str, Any]:
        """Detailed report: overall accuracy/macro-F1, per-class accuracy, and confusion matrix.

        The confusion matrix is row=true, col=pred, indexed by ``model.labels`` order.
        """
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

        y_true, y_pred = self._predict_true(dataset)
        labels = self.model.labels
        n = len(labels) if labels is not None else int(max(y_true.max(), y_pred.max(), 0)) + 1
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        row_totals = cm.sum(axis=1)
        names = list(labels) if labels is not None else list(range(n))
        per_class_acc = {
            names[i]: (float(cm[i, i] / row_totals[i]) if row_totals[i] else 0.0) for i in range(n)
        }
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "per_class_accuracy": per_class_acc,
            "labels": names,
            "confusion_matrix": cm.tolist(),
        }

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        raise NotImplementedError(
            "Hub upload is not implemented in 0.1.0. Use `model.save_pretrained(dir)` and upload manually."
        )
