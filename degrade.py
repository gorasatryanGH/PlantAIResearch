"""
degrade.py
Automated image-quality degradation for evaluating the robustness of plant
leaf disease classifiers.

Five corruption types are implemented, chosen to represent the dominant
failure modes of smartphone photography (see the paper's Methods section
for the full selection rationale):
    - gaussian_blur      (blur from camera shake / out-of-focus capture)
    - gaussian_noise      (sensor noise, typical under low light)
    - brightness_down     (insufficient ambient lighting)
    - jpeg_compression    (artifacts from strong JPEG compression)
    - downscale_upscale   (loss of detail from low sensor resolution or
                            aggressive image resizing)

Each corruption has 5 severity levels (severity 1..5), deliberately
matching the ImageNet-C protocol (Hendrycks & Dietterich, 2019) so results
are methodologically comparable to the standard robustness-evaluation
practice used elsewhere in computer vision.

IMPORTANT (experimental invariant): corruption is applied ONLY to the
train/val/test images that will be used for EVALUATION, never to images
used for training the "clean" model (i.e., the model is trained on clean
data and evaluated on corrupted data). Do not call these functions inside
training-time augmentations.

CLI usage:
    python degrade.py --input data/raw/test --output data/degraded \
        --corruptions gaussian_blur gaussian_noise jpeg_compression \
        --severities 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Individual corruption functions.
# All take a np.ndarray uint8 HxWx3 (RGB) image and a severity level 1..5,
# and return a np.ndarray uint8 image of the same size.
# ---------------------------------------------------------------------------

def gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Gaussian blur. Kernel size grows with severity."""
    sigma_levels = [0.6, 1.2, 2.0, 3.0, 4.5][severity - 1]
    ksize = int(2 * round(3 * sigma_levels) + 1)  # odd kernel size
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma_levels)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """Additive Gaussian noise (typical of low-light / small-sensor capture)."""
    sigma_levels = [0.03, 0.06, 0.10, 0.15, 0.22][severity - 1]
    x = img.astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma_levels, size=x.shape).astype(np.float32)
    out = np.clip(x + noise, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def brightness_down(img: np.ndarray, severity: int) -> np.ndarray:
    """Reduced brightness (insufficient lighting), with a mild associated contrast reduction."""
    factor_levels = [0.85, 0.70, 0.55, 0.40, 0.25][severity - 1]
    x = img.astype(np.float32) / 255.0
    # Brightness is reduced via the HSV color space so hue is not distorted.
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] *= factor_levels
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    """JPEG artifacts via a real encode/decode round-trip (quality decreases with severity)."""
    quality_levels = [80, 60, 40, 25, 12][severity - 1]
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality_levels)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def downscale_upscale(img: np.ndarray, severity: int) -> np.ndarray:
    """Resolution loss: downscale by a factor of N and upscale back (bilinear)."""
    factor_levels = [1.5, 2.0, 3.0, 4.0, 6.0][severity - 1]
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(w / factor_levels)), max(1, int(h / factor_levels))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "brightness_down": brightness_down,
    "jpeg_compression": jpeg_compression,
    "downscale_upscale": downscale_upscale,
}

SEVERITIES = [1, 2, 3, 4, 5]


def apply_corruption(img: np.ndarray, corruption_name: str, severity: int) -> np.ndarray:
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}. Available: {list(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return CORRUPTIONS[corruption_name](img, severity)


# ---------------------------------------------------------------------------
# Batch processing of an ImageFolder dataset
# ---------------------------------------------------------------------------

def degrade_dataset(
    input_dir: Path,
    output_dir: Path,
    corruptions: list[str],
    severities: list[int],
    seed: int = 42,
) -> None:
    """Walks input_dir/<class_name>/*.jpg and writes corrupted copies to
    output_dir/<corruption>/severity_<s>/<class_name>/<image>.
    """
    np.random.seed(seed)
    input_dir = Path(input_dir)
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subfolders found in {input_dir}. Expected an ImageFolder "
            f"layout: input_dir/<class_name>/<image>.jpg."
        )

    image_paths = []
    for cdir in class_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(cdir.glob(ext))

    print(f"Found {len(image_paths)} images across {len(class_dirs)} classes.")

    for corruption_name in corruptions:
        for severity in severities:
            out_root = output_dir / corruption_name / f"severity_{severity}"
            desc = f"{corruption_name} | severity {severity}"
            for img_path in tqdm(image_paths, desc=desc):
                class_name = img_path.parent.name
                out_dir = out_root / class_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / img_path.name
                if out_path.exists():
                    continue  # idempotent: re-running does not recompute existing files

                img = np.array(Image.open(img_path).convert("RGB"))
                degraded = apply_corruption(img, corruption_name, severity)
                Image.fromarray(degraded).save(out_path, quality=95)

    print(f"Done. Corrupted images saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply synthetic image-quality degradation to plant leaf photos")
    parser.add_argument("--input", type=str, required=True, help="Path to the clean val/test set (ImageFolder layout)")
    parser.add_argument("--output", type=str, default="data/degraded", help="Where to save corrupted copies")
    parser.add_argument(
        "--corruptions", nargs="+", default=list(CORRUPTIONS.keys()),
        choices=list(CORRUPTIONS.keys()), help="Which corruption types to apply",
    )
    parser.add_argument("--severities", nargs="+", type=int, default=SEVERITIES, choices=SEVERITIES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    degrade_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        corruptions=args.corruptions,
        severities=args.severities,
        seed=args.seed,
    )
