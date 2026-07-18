"""
prepare_data.py
Downloads the PlantVillage dataset and materializes it in ImageFolder
format (data/raw/{train,val,test}/<class_name>/*.jpg), as expected by
train.py and evaluate.py.

NOTE (implementation history): an earlier version of this script used
`datasets.load_dataset("mohanty/PlantVillage", "color")`. This is the
officially documented approach, but in practice it stopped working:
current versions of the Hugging Face `datasets` library no longer execute
a dataset repository's custom loading script (for security reasons), which
made only an empty auto-generated 'default' config visible. This version
avoids the problem entirely by downloading the dataset's files directly
via `huggingface_hub`, without depending on `datasets` at all.

Data source (verified manually against the repository file listing):
  - splits/color_train.txt, splits/color_test.txt — the dataset authors'
    official lists of relative file paths, split with "leaf grouping"
    taken into account (photographs of the same physical leaf never end up
    in both train and test);
  - data.zip (~2.2 GB) — an archive containing all images (color, grayscale,
    and segmented versions); paths inside match the lines in splits/*.txt.

This script does NOT require a GPU and does not consume Colab/Kaggle GPU
quota — it can safely be run on a CPU-only runtime while waiting for GPU
availability.

Usage:
    python prepare_data.py --config color --val_fraction 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_split_file(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stratified_train_val_split(
    train_paths: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Splits the official train set into train/val, stratified by class
    (the class is inferred from the second path component:
    raw/<config>/<class>/...).
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = {}
    for p in train_paths:
        class_name = p.split("/")[2]
        by_class.setdefault(class_name, []).append(p)

    train_final, val_final = [], []
    for class_name, items in by_class.items():
        items = items.copy()
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_final.extend(items[:n_val])
        train_final.extend(items[n_val:])
    return train_final, val_final


def extract_group(zf: zipfile.ZipFile, paths: list[str], out_dir: Path, split_name: str) -> None:
    for p in tqdm(paths, desc=f"Extracting {split_name}"):
        parts = p.split("/")
        class_name = parts[2]
        filename = parts[-1]
        dest_dir = out_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        if dest_path.exists():
            continue  # idempotent: re-running does not redo completed work
        with zf.open(p) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())


def main(config: str, val_fraction: float, seed: int, out_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Could not find the 'huggingface_hub' library. Install it with: pip install huggingface_hub"
        ) from e

    repo_id = "mohanty/PlantVillage"

    print(f"Downloading the official train/test split lists (config={config})...")
    train_split_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=f"splits/{config}_train.txt"
    )
    test_split_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=f"splits/{config}_test.txt"
    )

    train_paths_all = read_split_file(train_split_path)
    test_paths = read_split_file(test_split_path)
    print(f"Official train: {len(train_paths_all)} files | test: {len(test_paths)} files")

    class_names = sorted({p.split("/")[2] for p in train_paths_all})
    print(f"Classes: {len(class_names)}")

    train_paths, val_paths = stratified_train_val_split(train_paths_all, val_fraction, seed)
    print(f"After carving out val: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    print("Downloading the image archive (data.zip, ~2.2 GB, one-time download, cached afterwards)...")
    zip_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data.zip")

    out_dir = Path(out_dir)
    print("Extracting the required files into data/raw/{train,val,test}/...")
    with zipfile.ZipFile(zip_path) as zf:
        # Quick sanity check that paths from the split files actually exist in the archive.
        sample_check = train_paths[:5] + test_paths[:5]
        archive_names = set(zf.namelist())
        missing = [p for p in sample_check if p not in archive_names]
        if missing:
            raise RuntimeError(
                f"Paths from the split files were not found in data.zip (example: {missing[:3]}). "
                f"The archive layout may have changed — please report this error."
            )

        extract_group(zf, train_paths, out_dir, "train")
        extract_group(zf, val_paths, out_dir, "val")
        extract_group(zf, test_paths, out_dir, "test")

    (out_dir / "class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    print(f"\nDone. Data has been laid out under {out_dir}/{{train,val,test}}/<class_name>/*.jpg")
    print(f"Class name list saved to {out_dir / 'class_names.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare the PlantVillage dataset")
    parser.add_argument("--config", type=str, default="color", choices=["color", "grayscale", "segmented"])
    parser.add_argument("--val_fraction", type=float, default=0.15,
                         help="Fraction of the original train set to carve out as val (stratified)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config=args.config, val_fraction=args.val_fraction, seed=args.seed, out_dir=Path(args.out_dir))
