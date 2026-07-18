"""
evaluate.py
Evaluates a trained model on the clean and/or corrupted test sets.
Saves:
    results/metrics_<model>_<condition>.json   — full metrics + confusion matrix
    results/summary.csv                         — summary table across all runs
                                                    (used for the paper's tables/figures)

Examples:

  # evaluate on the clean test set:
  python evaluate.py --checkpoint models/checkpoints/resnet50_best.pt \
      --data_dir data/raw/test --condition clean

  # evaluate on all corrupted versions (produced beforehand by degrade.py):
  python evaluate.py --checkpoint models/checkpoints/resnet50_best.pt \
      --degraded_root data/degraded --sweep_all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from utils import build_dataloader, compute_metrics, get_device, get_logger

logger = get_logger("evaluate")

SUMMARY_CSV_FIELDS = [
    "model", "corruption", "severity", "accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted", "n_samples",
]


def load_model(checkpoint_path: Path, device: torch.device):
    """Loads a checkpoint and rebuilds the architecture from its saved config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["config"]["model_name"]
    class_names = ckpt["class_names"]

    # Local import so train.py is not a hard dependency of this module.
    from train import build_model
    model, _ = build_model(model_name, num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, model_name


@torch.no_grad()
def evaluate_on_dir(model, data_dir: Path, class_names: list[str], device, image_size=224, batch_size=32):
    loader, dataset_classes = build_dataloader(
        data_dir, image_size=image_size, batch_size=batch_size, train=False,
    )
    if dataset_classes != class_names:
        logger.warning(
            "The class order in the test folder differs from the order used during training! "
            "Labels may be mismatched. Verify that the folder structure matches train exactly."
        )

    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    metrics = compute_metrics(all_targets, all_preds, class_names=class_names)
    metrics["n_samples"] = len(all_targets)
    return metrics


def append_to_summary(summary_path: Path, row: dict) -> None:
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_CSV_FIELDS})


def evaluate_condition(
    model, class_names, model_name, device, data_dir: Path,
    corruption: str, severity, results_dir: Path, image_size=224, batch_size=32,
):
    logger.info(f"[{model_name}] Evaluating: corruption={corruption} severity={severity} dir={data_dir}")
    metrics = evaluate_on_dir(model, data_dir, class_names, device, image_size, batch_size)

    tag = f"{corruption}_sev{severity}" if severity is not None else corruption
    out_path = results_dir / f"metrics_{model_name}_{tag}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    logger.info(
        f"  accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
        f"(n={metrics['n_samples']}) -> {out_path}"
    )

    append_to_summary(results_dir / "summary.csv", {
        "model": model_name,
        "corruption": corruption,
        "severity": severity if severity is not None else "",
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "n_samples": metrics["n_samples"],
    })
    return metrics


def sweep_all_degradations(
    model, class_names, model_name, device, degraded_root: Path,
    results_dir: Path, image_size=224, batch_size=32,
):
    """Walks all of data/degraded/<corruption>/severity_<s>/ and evaluates the model on each."""
    degraded_root = Path(degraded_root)
    if not degraded_root.exists():
        raise FileNotFoundError(
            f"{degraded_root} does not exist. Run degrade.py first to generate corrupted versions."
        )
    for corruption_dir in sorted(degraded_root.iterdir()):
        if not corruption_dir.is_dir():
            continue
        for severity_dir in sorted(corruption_dir.iterdir()):
            if not severity_dir.is_dir():
                continue
            severity = severity_dir.name.replace("severity_", "")
            evaluate_condition(
                model, class_names, model_name, device, severity_dir,
                corruption=corruption_dir.name, severity=int(severity),
                results_dir=results_dir, image_size=image_size, batch_size=batch_size,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on clean/corrupted data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                         help="ImageFolder directory for a single evaluation run (e.g. the clean test set)")
    parser.add_argument("--condition", type=str, default="clean",
                         help="Name of the condition for a single evaluation run (e.g. 'clean')")
    parser.add_argument("--degraded_root", type=str, default="data/degraded")
    parser.add_argument("--sweep_all", action="store_true",
                         help="Evaluate on every corrupted version found under --degraded_root")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    logger.info(f"Device: {device}")

    model, class_names, model_name = load_model(Path(args.checkpoint), device)
    results_dir = Path(args.results_dir)

    if args.data_dir is not None:
        evaluate_condition(
            model, class_names, model_name, device, Path(args.data_dir),
            corruption=args.condition, severity=None, results_dir=results_dir,
            image_size=args.image_size, batch_size=args.batch_size,
        )

    if args.sweep_all:
        sweep_all_degradations(
            model, class_names, model_name, device, Path(args.degraded_root),
            results_dir=results_dir, image_size=args.image_size, batch_size=args.batch_size,
        )

    if args.data_dir is None and not args.sweep_all:
        logger.warning("Nothing to do: pass --data_dir for a single evaluation, or --sweep_all.")
