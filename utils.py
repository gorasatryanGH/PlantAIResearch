"""
utils.py
Shared helper functions used by train.py and evaluate.py:
- fixing the random seed for reproducibility;
- building DataLoaders from an ImageFolder-style directory structure;
- computing metrics (Accuracy, Precision, Recall, F1) and the confusion matrix;
- saving/loading the experiment configuration.

Expected data layout (ImageFolder format):
    data/raw/<split>/<class_name>/<image>.jpg
    data/degraded/<corruption>/severity_<n>/<split>/<class_name>/<image>.jpg

where split in {train, val, test}.
"""

from __future__ import annotations

import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fixes the seed for random, numpy, and torch (CPU + CUDA).

    NOTE: fully deterministic torch execution (torch.use_deterministic_algorithms)
    can noticeably slow down training for some layers (e.g. upsampling). This
    codebase uses a practical compromise: reproducibility of weight
    initialization and data ordering is guaranteed, but individual CUDA
    kernels may still introduce microscopic non-deterministic deviations.
    This is standard practice in the field and should not be treated as a
    methodological flaw, but it is explicitly disclosed here and in the
    paper's "Materials and Methods" section for transparency.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    model_name: str = "resnet50"          # resnet50 | efficientnet_b0 | convnext_tiny | vit_b_16
    data_dir: str = "data/raw"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 2       # epochs with a frozen backbone (linear probing before fine-tuning)
    seed: int = 42
    checkpoint_dir: str = "models/checkpoints"
    num_classes: Optional[int] = None      # inferred automatically from the dataset

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# ImageNet normalization statistics — used by all four architectures, since
# all backbones are pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Returns the torchvision transform pipeline for train or eval.

    IMPORTANT: the eval transform contains NO augmentation and NO artificial
    corruption — corruption is applied beforehand, as a separate offline
    step, by degrade.py. This is deliberate, for two reasons:
      1) corrupted images can be inspected visually and reused across runs;
      2) it avoids the risk of accidentally "augmenting" the training set
         with the same corruptions used to evaluate robustness, which would
         bias the evaluation optimistically (the model would effectively be
         trained on the exact perturbations it is later tested against).
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloader(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    train: bool = False,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> tuple[DataLoader, list[str]]:
    """Builds a DataLoader from an ImageFolder directory root/<class_name>/*.jpg.

    Returns (dataloader, list_of_class_names_in_index_order).
    """
    tfm = build_transforms(image_size=image_size, train=train)
    dataset = datasets.ImageFolder(root=str(root), transform=tfm)
    if shuffle is None:
        shuffle = train
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset.classes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, class_names: Optional[list[str]] = None) -> dict:
    """Computes Accuracy, macro/weighted Precision/Recall/F1, and the confusion matrix.

    Macro-averaging is used as the primary metric (all classes are treated
    as equally important, which is appropriate given possible class
    imbalance among disease categories), while weighted-averaging is also
    retained for a complete report.
    """
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class = None
    if class_names is not None:
        p_c, r_c, f1_c, support_c = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0,
            labels=list(range(len(class_names))),
        )
        per_class = {
            name: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f1_c[i]),
                "support": int(support_c[i]),
            }
            for i, name in enumerate(class_names)
        }

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
