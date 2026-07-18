"""
train.py
Fine-tunes one of four architectures on the CLEAN (non-corrupted)
PlantVillage images.

Supported architectures (--model):
    resnet50, efficientnet_b0, convnext_tiny, vit_b_16

Training strategy:
    1) for the first `freeze_backbone_epochs` epochs, the backbone is
       frozen and only the new classification head is trained (linear
       probing);
    2) afterwards, the backbone is fully unfrozen and fine-tuned at a
       lower learning rate until `epochs` is reached.
This is a standard, reproducible practice for comparing pretrained
architectures on a moderately sized dataset, and it reduces the risk of
overfitting in the early epochs.

Example:
    python train.py --model resnet50 --data_dir data/raw --epochs 15
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import (
    ExperimentConfig, set_seed, get_device, build_dataloader,
    compute_metrics, get_logger,
)

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int) -> tuple[nn.Module, list[nn.Parameter]]:
    """Returns (model, list of classification-head parameters).

    All backbones are initialized with ImageNet-pretrained weights
    (torchvision.models, weights="IMAGENET1K_*"); the final layer is
    replaced to match num_classes.
    """
    if model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_params = list(model.fc.parameters())

    elif model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head_params = list(model.classifier[-1].parameters())

    elif model_name == "convnext_tiny":
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head_params = list(model.classifier[-1].parameters())

    elif model_name == "vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        head_params = list(model.heads.head.parameters())

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: resnet50, efficientnet_b0, convnext_tiny, vit_b_16"
        )

    return model, head_params


def set_backbone_trainable(model: nn.Module, head_params: list[nn.Parameter], trainable: bool) -> None:
    """Freezes/unfreezes all parameters except the classification head."""
    head_ids = {id(p) for p in head_params}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, train: bool) -> dict:
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_model(cfg: ExperimentConfig) -> None:
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    train_loader, class_names = build_dataloader(
        Path(cfg.data_dir) / "train", image_size=cfg.image_size,
        batch_size=cfg.batch_size, train=True, num_workers=cfg.num_workers,
    )
    val_loader, _ = build_dataloader(
        Path(cfg.data_dir) / "val", image_size=cfg.image_size,
        batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers,
    )
    cfg.num_classes = len(class_names)
    logger.info(f"Classes: {cfg.num_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model, head_params = build_model(cfg.model_name, cfg.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val_f1 = -1.0
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{cfg.model_name}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        # Phase 1: linear probing (frozen backbone)
        frozen_phase = epoch <= cfg.freeze_backbone_epochs
        set_backbone_trainable(model, head_params, trainable=not frozen_phase)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        lr = cfg.lr if frozen_phase else cfg.lr / 10  # lower LR during full fine-tuning
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=cfg.weight_decay)

        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, None, criterion, device, train=False)
        dt = time.time() - t0

        phase_str = "frozen-backbone" if frozen_phase else "fine-tuning"
        logger.info(
            f"[{cfg.model_name}] epoch {epoch}/{cfg.epochs} ({phase_str}, {dt:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "config": cfg.__dict__,
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, checkpoint_path)
            logger.info(f"  -> new best checkpoint saved: {checkpoint_path} (val_f1_macro={best_val_f1:.4f})")

    cfg.save(checkpoint_dir / f"{cfg.model_name}_config.json")
    logger.info(f"Training of {cfg.model_name} finished. Best val_f1_macro={best_val_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune plant disease classification models")
    parser.add_argument("--model", type=str, required=True,
                         choices=["resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--data_dir", type=str, default="data/raw",
                         help="Expects data_dir/train and data_dir/val in ImageFolder format")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExperimentConfig(
        model_name=args.model,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )
    train_model(cfg)
