# ============================================================
# PREPROCESSING — Chest X-Ray Diagnosis ML System
# ============================================================
# Medical imaging preprocessing:
#   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     for X-ray contrast enhancement
#   - Normalization with ImageNet stats (transfer learning)
#   - Train / val / test transform pipelines
#   - Batch normalizer for computed statistics
# ============================================================

import numpy as np
import logging
from typing import Tuple, Optional, List

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from src.config import IMAGE_SIZE, PATHOLOGY_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)

# ImageNet statistics — standard for pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# CLAHE TRANSFORM — contrast enhancement for X-rays
# ============================================================

class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Standard preprocessing step in radiology AI to enhance subtle findings
    like nodules and infiltrates that are low-contrast in raw X-rays.

    Works on PIL images; returns PIL image.
    clip_limit controls over-amplification — 2.0 is standard for chest X-rays.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit      = clip_limit
        self.tile_grid_size  = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            import cv2
            img_np  = np.array(img.convert("L"), dtype=np.uint8)
            clahe   = cv2.createCLAHE(
                clipLimit   = self.clip_limit,
                tileGridSize= (self.tile_grid_size, self.tile_grid_size)
            )
            enhanced = clahe.apply(img_np)
            # Convert back to RGB (3-channel for pretrained backbones)
            enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
            return Image.fromarray(enhanced_rgb)
        except ImportError:
            # cv2 not available — fall back to PIL equalization
            logger.debug("cv2 not available — using PIL ImageOps.equalize fallback")
            return ImageOps.equalize(img.convert("L")).convert("RGB")
        except Exception as e:
            logger.warning("CLAHE failed: %s — returning original", e)
            return img.convert("RGB")

    def __repr__(self) -> str:
        return f"CLAHETransform(clip_limit={self.clip_limit})"


# ============================================================
# SAFE RANDOM ROTATION  (bounded for medical images)
# ============================================================

class SafeRotation:
    """
    Bounded random rotation for chest X-rays.
    Medical constraint: >10° rotation distorts anatomy relationships.
    """

    def __init__(self, max_degrees: float = 7.0):
        self.max_degrees = max_degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = np.random.uniform(-self.max_degrees, self.max_degrees)
        return TF.rotate(img, angle, fill=0)   # fill=0 → black (air in X-ray)

    def __repr__(self) -> str:
        return f"SafeRotation(max_degrees={self.max_degrees})"


# ============================================================
# TRANSFORM PIPELINES
# ============================================================

def build_train_transform(use_clahe: bool = True) -> T.Compose:
    """
    Training augmentation pipeline for chest X-rays.

    Augmentations validated for chest X-ray AI:
      ✅ Horizontal flip    — left/right mirroring acceptable
      ✅ Small rotation     — < 10° keeps anatomy intact
      ✅ Brightness/contrast jitter — simulates scanner variation
      ✅ CLAHE             — enhance low-contrast findings
      ❌ Vertical flip     — inverts anatomy (NEVER for X-rays)
      ❌ Large crops       — can cut out pathology regions
      ❌ Color saturation  — X-rays are grayscale; irrelevant
    """
    steps = []

    if use_clahe:
        steps.append(CLAHETransform(clip_limit=2.0))

    steps += [
        T.Resize(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        SafeRotation(max_degrees=7.0),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return T.Compose(steps)


def build_eval_transform(use_clahe: bool = True) -> T.Compose:
    """
    Deterministic eval/test transform — no augmentation.
    Same CLAHE as training to keep preprocessing consistent.
    """
    steps = []

    if use_clahe:
        steps.append(CLAHETransform(clip_limit=2.0))

    steps += [
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return T.Compose(steps)


def build_tta_transforms(n_augments: int = 5) -> List[T.Compose]:
    """
    Test-Time Augmentation (TTA) transforms.
    Run inference N times with slight variations → average predictions.
    Improves AUC by ~1-2% on medical imaging tasks.
    """
    base = [CLAHETransform(clip_limit=2.0), T.Resize(IMAGE_SIZE)]
    norm = [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    augments = [
        [],                                               # original
        [T.RandomHorizontalFlip(p=1.0)],                  # h-flip
        [SafeRotation(max_degrees=5)],                    # slight rotate
        [T.ColorJitter(brightness=0.1, contrast=0.1)],    # jitter
        [T.RandomHorizontalFlip(p=1.0), SafeRotation(max_degrees=3)],  # flip+rotate
    ]

    transforms_list = []
    for aug in augments[:n_augments]:
        transforms_list.append(T.Compose(base + aug + norm))

    logger.info("TTA: %d augmentation transforms prepared", len(transforms_list))
    return transforms_list


# ============================================================
# DENORMALIZE  (for visualization / Grad-CAM display)
# ============================================================

def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.
    Input:  (C, H, W) normalized tensor
    Output: (C, H, W) tensor with values in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=tensor.dtype).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# ============================================================
# LABEL PREPROCESSING UTILITIES
# ============================================================

def binarize_predictions(
    probs:      np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply per-class thresholds to probability matrix.

    Args:
        probs:      (N, 14) float array of predicted probabilities
        thresholds: (14,) array of per-class thresholds
                    If None, uses 0.5 for all classes.
    Returns:
        (N, 14) int binary predictions
    """
    if thresholds is None:
        thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

    return (probs >= thresholds).astype(int)


def build_threshold_array(threshold_dict: dict) -> np.ndarray:
    """
    Convert per-class threshold dict → ordered numpy array.
    Fills missing labels with the 'default' key or 0.5.
    """
    default = threshold_dict.get("default", 0.5)
    return np.array([
        threshold_dict.get(lbl, default)
        for lbl in PATHOLOGY_LABELS
    ], dtype=np.float32)


# ============================================================
# BATCH STATISTICS — for drift monitoring
# ============================================================

def compute_image_statistics(images_dir: str, sample_n: int = 1000) -> dict:
    """
    Compute pixel-level statistics from a sample of X-ray images.
    Used as reference distribution for PSI drift detection.
    """
    import os
    from glob import glob

    image_files = glob(os.path.join(images_dir, "*.png"))[:sample_n]
    if not image_files:
        logger.warning("No images found in %s for statistics", images_dir)
        return {}

    means, stds = [], []
    for f in image_files:
        try:
            arr  = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
            means.append(arr.mean())
            stds.append(arr.std())
        except Exception:
            continue

    stats = {
        "pixel_mean_avg": float(np.mean(means)),
        "pixel_mean_std": float(np.std(means)),
        "pixel_std_avg":  float(np.mean(stds)),
        "n_sampled":      len(means),
    }

    logger.info("Image statistics  |  pixel_mean=%.4f  pixel_std=%.4f  n=%d",
                stats["pixel_mean_avg"], stats["pixel_std_avg"], len(means))
    return stats
