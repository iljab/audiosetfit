"""`AudioSetFitModel` and `AudioSetFitHead`.

Mirrors `setfit.SetFitModel`: a body (`AudioEncoder`) that produces embeddings plus a
classifier head that is either an sklearn `LogisticRegression` (default) or a
differentiable torch `AudioSetFitHead`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import trange

from .encoders import AudioEncoder, build_encoder

CONFIG_NAME = "config_audiosetfit.json"
HEAD_NAME = "model_head"  # extension added depending on head type
BACKBONE_SUBDIR = "backbone"


class AudioSetFitHead(nn.Module):
    """Differentiable classification head (linear + temperature-scaled softmax/sigmoid)."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 2,
        temperature: float = 1.0,
        eps: float = 1e-5,
        bias: bool = True,
        multitarget: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if out_features == 1:
            out_features = 2
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.eps = eps
        self.multitarget = multitarget
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None):
        temperature = temperature or self.temperature
        x = x.to(self.linear.weight.device)
        logits = self.linear(x) / (temperature + self.eps)
        probs = torch.sigmoid(logits) if self.multitarget else nn.functional.softmax(logits, dim=-1)
        return logits, probs

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)[1]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        if self.multitarget:
            return torch.where(probs >= 0.5, 1, 0)
        return torch.argmax(probs, dim=-1)

    def get_loss_fn(self) -> nn.Module:
        return nn.BCEWithLogitsLoss() if self.multitarget else nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_config_dict(self) -> Dict:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "temperature": self.temperature,
            "multitarget": self.multitarget,
        }


class AudioSetFitModel:
    """A few-shot audio classifier: an `AudioEncoder` body + classification head."""

    def __init__(
        self,
        model_body: AudioEncoder,
        model_head: Union["AudioSetFitHead", "object"],
        labels: Optional[List[str]] = None,
        normalize_embeddings: bool = True,
        multi_target_strategy: Optional[str] = None,
    ) -> None:
        self.model_body = model_body
        self.model_head = model_head
        self.labels = labels
        self.normalize_embeddings = normalize_embeddings
        self.multi_target_strategy = multi_target_strategy

    # ------------------------------------------------------------------ properties
    @property
    def has_differentiable_head(self) -> bool:
        return isinstance(self.model_head, nn.Module)

    @property
    def device(self) -> torch.device:
        return self.model_body.device

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(enumerate(self.labels)) if self.labels else {}

    @property
    def label2id(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)} if self.labels else {}

    # ------------------------------------------------------------------ encoding
    def encode(self, inputs, batch_size: int = 16, show_progress_bar: Optional[bool] = None):
        return self.model_body.encode(
            inputs,
            batch_size=batch_size,
            normalize=self.normalize_embeddings,
            convert_to_tensor=self.has_differentiable_head,
            show_progress_bar=show_progress_bar,
        )

    # ------------------------------------------------------------------ head training
    def fit(
        self,
        x_train: List,
        y_train: List,
        num_epochs: int = 25,
        batch_size: int = 16,
        head_learning_rate: float = 1e-2,
        l2_weight: float = 0.01,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> None:
        embeddings = self.encode(x_train, batch_size=batch_size, show_progress_bar=show_progress_bar)
        if self.has_differentiable_head:
            self._fit_torch_head(embeddings, y_train, num_epochs, batch_size, head_learning_rate, l2_weight, show_progress_bar)
        else:
            self.model_head.fit(embeddings, list(y_train))
            if self.labels is None and self.multi_target_strategy is None:
                try:
                    classes = self.model_head.classes_
                    if classes.dtype.char == "U":
                        self.labels = classes.tolist()
                except Exception:
                    pass

    def _fit_torch_head(self, embeddings, y_train, num_epochs, batch_size, lr, l2_weight, show_progress_bar) -> None:
        head = self.model_head
        head.train()
        device = head.device
        x = embeddings if isinstance(embeddings, torch.Tensor) else torch.as_tensor(np.asarray(embeddings))
        x = x.to(device).float()
        y = torch.as_tensor(np.asarray(y_train))
        y = y.float().to(device) if head.multitarget else y.long().to(device)
        criterion = head.get_loss_fn()
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=l2_weight)
        n = x.shape[0]
        for _ in trange(num_epochs, desc="Head epoch", disable=not show_progress_bar):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                logits, _ = head(x[idx])
                loss = criterion(logits, y[idx])
                loss.backward()
                optimizer.step()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_proba(self, inputs, batch_size: int = 16, show_progress_bar: Optional[bool] = None):
        is_singular = not isinstance(inputs, list)
        if is_singular:
            inputs = [inputs]
        embeddings = self.encode(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar)
        probs = self.model_head.predict_proba(embeddings)
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        return probs[0] if is_singular else probs

    @torch.no_grad()
    def predict(self, inputs, batch_size: int = 16, use_labels: bool = True, show_progress_bar: Optional[bool] = None):
        is_singular = not isinstance(inputs, list)
        if is_singular:
            inputs = [inputs]
        embeddings = self.encode(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar)
        preds = self.model_head.predict(embeddings)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        preds = np.asarray(preds)
        if use_labels and self.labels and preds.ndim == 1 and preds.dtype.char != "U":
            outputs = [self.labels[int(p)] for p in preds]
        else:
            outputs = preds.tolist()
        return outputs[0] if is_singular else outputs

    def __call__(self, inputs, **kwargs):
        return self.predict(inputs, **kwargs)

    # ------------------------------------------------------------------ freeze/move
    def freeze(self, component: Optional[Literal["body", "head"]] = None) -> None:
        if component in (None, "body"):
            for p in self.model_body.parameters():
                p.requires_grad = False
        if component in (None, "head") and self.has_differentiable_head:
            for p in self.model_head.parameters():
                p.requires_grad = False

    def unfreeze(self, component: Optional[Literal["body", "head"]] = None) -> None:
        if component in (None, "body"):
            for p in self.model_body.parameters():
                p.requires_grad = True
        if component in (None, "head") and self.has_differentiable_head:
            for p in self.model_head.parameters():
                p.requires_grad = True

    def to(self, device: Union[str, torch.device]) -> "AudioSetFitModel":
        self.model_body.to(device)
        if self.has_differentiable_head:
            self.model_head.to(device)
        return self

    # ------------------------------------------------------------------ persistence
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        backbone_dir = save_directory / BACKBONE_SUBDIR
        self.model_body.save(str(backbone_dir))

        head_is_torch = self.has_differentiable_head
        if head_is_torch:
            head_path = save_directory / f"{HEAD_NAME}.pt"
            torch.save(
                {"state_dict": self.model_head.to("cpu").state_dict(), "config": self.model_head.get_config_dict()},
                head_path,
            )
            self.model_head.to(self.device)
        else:
            import joblib

            head_path = save_directory / f"{HEAD_NAME}.pkl"
            joblib.dump(self.model_head, head_path)

        config = {
            "labels": self.labels,
            "normalize_embeddings": self.normalize_embeddings,
            "multi_target_strategy": self.multi_target_strategy,
            "head_type": "torch" if head_is_torch else "sklearn",
            "embedding_dim": self.model_body.embedding_dim,
            "backbone_model_id": self.model_body.model_id,
        }
        with open(save_directory / CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        labels: Optional[List[str]] = None,
        use_differentiable_head: bool = False,
        head_params: Optional[Dict] = None,
        multi_target_strategy: Optional[str] = None,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        encoder_type: Optional[str] = None,
        **kwargs,
    ) -> "AudioSetFitModel":
        """Load a saved audiosetfit model directory, or initialize a fresh model from a backbone id.

        Args:
            encoder_type: force a specific encoder from the registry instead of auto-detecting
                from the backbone's ``model_type``.
        """
        config_path = os.path.join(model_id, CONFIG_NAME) if os.path.isdir(model_id) else None

        if config_path and os.path.isfile(config_path):
            return cls._load_saved(model_id, device=device)

        # Fresh model from a backbone (hub id or local backbone dir)
        model_body = build_encoder(model_id, device=device, encoder_type=encoder_type)
        head_params = head_params or {}
        if use_differentiable_head:
            out_features = len(labels) if labels else head_params.pop("out_features", 2)
            model_head = AudioSetFitHead(
                in_features=model_body.embedding_dim,
                out_features=out_features,
                multitarget=multi_target_strategy is not None,
                device=str(model_body.device),
                **head_params,
            )
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

            clf = LogisticRegression(**head_params)
            if multi_target_strategy == "one-vs-rest":
                model_head = OneVsRestClassifier(clf)
            elif multi_target_strategy == "multi-output":
                model_head = MultiOutputClassifier(clf)
            elif multi_target_strategy == "classifier-chain":
                model_head = ClassifierChain(clf)
            else:
                model_head = clf

        return cls(
            model_body=model_body,
            model_head=model_head,
            labels=labels,
            normalize_embeddings=normalize_embeddings,
            multi_target_strategy=multi_target_strategy,
        )

    @classmethod
    def _load_saved(cls, save_directory: str, device: Optional[str] = None) -> "AudioSetFitModel":
        with open(os.path.join(save_directory, CONFIG_NAME), "r", encoding="utf-8") as f:
            config = json.load(f)

        backbone_dir = os.path.join(save_directory, BACKBONE_SUBDIR)
        model_body = build_encoder(backbone_dir, device=device)

        if config["head_type"] == "torch":
            ckpt = torch.load(os.path.join(save_directory, f"{HEAD_NAME}.pt"), map_location="cpu")
            model_head = AudioSetFitHead(device=str(model_body.device), **ckpt["config"])
            model_head.load_state_dict(ckpt["state_dict"])
            model_head.to(model_body.device)
        else:
            import joblib

            model_head = joblib.load(os.path.join(save_directory, f"{HEAD_NAME}.pkl"))

        return cls(
            model_body=model_body,
            model_head=model_head,
            labels=config.get("labels"),
            normalize_embeddings=config.get("normalize_embeddings", True),
            multi_target_strategy=config.get("multi_target_strategy"),
        )
