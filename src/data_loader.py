# ============================================================
# DATA LOADER — Chest X-Ray Diagnosis ML System
# ============================================================
# NIH ChestX-ray14 dataset:
#   - 112,120 X-ray images from 30,805 unique patients
#   - 14 disease labels (multi-label: one image can have many)
#   - Labels are machine-generated NLP from radiology reports
#   - Data_Entry_2017.csv: Image Index, Finding Labels, etc.
# ============================================================

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from src.config import (
    PATHOLOGY_LABELS, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    NUM_WORKERS, RANDOM_STATE, DATA_DIR, IMAGES_DIR, LABELS_CSV
)

logger = logging.getLogger(__name__)


# ============================================================
# LABEL UTILITIES
# ============================================================

def parse_finding_labels(label_str: str) -> List[str]:
    """
    Parse pipe-separated finding labels from Data_Entry_2017.csv.
    e.g. 'Atelectasis|Infiltration' → ['Atelectasis', 'Infiltration']
    'No Finding' → []
    """
    if pd.isna(label_str) or label_str.strip() == "No Finding":
        return []
    return [lbl.strip() for lbl in label_str.split("|") if lbl.strip()]


def labels_to_vector(labels: List[str]) -> np.ndarray:
    """
    Convert list of finding labels → binary vector of length 14.
    e.g. ['Pneumonia', 'Effusion'] → [0,0,1,0,0,0,1,0,0,0,0,0,0,0]
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    label_idx = {lbl: i for i, lbl in enumerate(PATHOLOGY_LABELS)}
    for lbl in labels:
        if lbl in label_idx:
            vec[label_idx[lbl]] = 1.0
    return vec


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize NIH ChestX-ray14 metadata CSV.
    Expected columns from Data_Entry_2017.csv:
        Image Index, Finding Labels, Follow-up #, Patient ID,
        Patient Age, Patient Gender, View Position,
        OriginalImage[Width Height], OriginalImagePixelSpacing[x y]
    """
    # ── Normalize column names ────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[\[\] ]+", "_", regex=True)
        .str.lower()
    )

    # ── Required columns ──────────────────────────────────────
    required = ["image_index", "finding_labels"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Deduplication ──────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["image_index"], ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate image entries", dropped)

    # ── Null check ────────────────────────────────────────────
    nulls = df[["image_index", "finding_labels"]].isnull().sum().sum()
    if nulls > 0:
        logger.warning("%d nulls in image_index / finding_labels — dropping", nulls)
        df.dropna(subset=["image_index", "finding_labels"], inplace=True)

    # ── Minimum size check ────────────────────────────────────
    if len(df) < 100:
        raise ValueError(f"Dataset too small: {len(df)} rows")

    logger.info(
        "Metadata validation passed  |  total_images=%d", len(df)
    )
    return df


# ============================================================
# LABEL STATISTICS
# ============================================================

def compute_label_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute prevalence (positive rate) per label.
    Used for weighted loss and class imbalance reporting.
    """
    stats = {}
    for lbl in PATHOLOGY_LABELS:
        count = df["finding_labels"].apply(
            lambda x: lbl in parse_finding_labels(str(x))
        ).sum()
        stats[lbl] = round(count / len(df), 4)

    logger.info("Label prevalence computed:")
    for lbl, rate in stats.items():
        logger.info("  %-22s  %.2f%%", lbl, rate * 100)

    return stats


def compute_pos_weights(df: pd.DataFrame, scale: float = 2.0) -> torch.Tensor:
    """
    Compute class-balance pos_weight for BCEWithLogitsLoss.
    pos_weight[i] = (neg_count / pos_count) * scale
    Handles zero positives gracefully.
    """
    weights = []
    for lbl in PATHOLOGY_LABELS:
        pos = df["finding_labels"].apply(
            lambda x: lbl in parse_finding_labels(str(x))
        ).sum()
        neg = len(df) - pos
        w   = (neg / max(pos, 1)) * scale
        weights.append(w)
        logger.debug("pos_weight[%s] = %.2f", lbl, w)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# TRAIN / VAL / TEST SPLIT  (patient-level)
# ============================================================

def patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CRITICAL: Split must be at PATIENT level, not image level.
    A patient can appear in many images — splitting at image level
    leaks patient-specific features into validation/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # ── Detect patient ID column ──────────────────────────────
    pid_col = None
    for candidate in ["patient_id", "patient_id_", "patientid"]:
        if candidate in df.columns:
            pid_col = candidate
            break

    if pid_col is None:
        # NIH data: Image Index is "00000001_000.png" → first 8 digits = patient
        logger.warning("Patient ID column not found — inferring from image index")
        df = df.copy()
        df["patient_id_inferred"] = df["image_index"].apply(
            lambda x: str(x).split("_")[0]
        )
        pid_col = "patient_id_inferred"

    patients = df[pid_col].unique()
    rng      = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(patients)

    n       = len(patients)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train:n_train + n_val])
    test_patients  = set(patients[n_train + n_val:])

    df_train = df[df[pid_col].isin(train_patients)].reset_index(drop=True)
    df_val   = df[df[pid_col].isin(val_patients)].reset_index(drop=True)
    df_test  = df[df[pid_col].isin(test_patients)].reset_index(drop=True)

    logger.info(
        "Patient-level split  |  train=%d imgs (%d pts)  val=%d imgs (%d pts)  test=%d imgs (%d pts)",
        len(df_train), len(train_patients),
        len(df_val),   len(val_patients),
        len(df_test),  len(test_patients),
    )

    return df_train, df_val, df_test


# ============================================================
# PYTORCH DATASET
# ============================================================

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.

    Supports:
        - Multi-label binary targets (14 classes)
        - On-the-fly augmentation for training
        - Graceful handling of missing images (returns black image)
        - Metadata pass-through for explainability
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        images_dir:  str,
        transform:   Optional[object] = None,
        is_train:    bool = False,
    ):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform
        self.is_train   = is_train
        self.label_idx  = {lbl: i for i, lbl in enumerate(PATHOLOGY_LABELS)}

        # ── Pre-compute label vectors for speed ──────────────
        self.labels = np.stack([
            labels_to_vector(parse_finding_labels(str(row["finding_labels"])))
            for _, row in self.df.iterrows()
        ])

        logger.info(
            "Dataset initialized  |  split=%s  n=%d",
            "train" if is_train else "eval", len(self.df)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        # NIH subfolders: images_001/images/, images_002/images/ ...
        image_name = row["image_index"]
        image_path = os.path.join(self.images_dir, image_name)  # direct try

        if not os.path.exists(image_path):
            for i in range(1, 13):
                candidate = os.path.join(
                    self.images_dir, f"images_{i:03d}", "images", image_name
                )
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        # ── Load image ────────────────────────────────────────
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            logger.warning("Missing image: %s — using black fallback", image_path)
            img = Image.fromarray(
                np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
            )

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img, label, row["image_index"]


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms(is_train: bool = False) -> transforms.Compose:
    """
    Training: augmentation appropriate for chest X-rays.
    Val/Test: deterministic resize + normalize only.

    Augmentations chosen for medical imaging:
        - Horizontal flip (left-right mirror is valid for chest)
        - Small random rotation (< 10°, avoid distorting anatomy)
        - Small brightness/contrast jitter
        - NO vertical flip (would invert anatomy — incorrect)
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats (fine for X-rays)
    std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloaders(
    df_train:   pd.DataFrame,
    df_val:     pd.DataFrame,
    df_test:    pd.DataFrame,
    images_dir: str = IMAGES_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders.
    Train loader uses WeightedRandomSampler to handle class imbalance
    at the batch level.
    """

    train_dataset = ChestXrayDataset(df_train, images_dir, get_transforms(True),  is_train=True)
    val_dataset   = ChestXrayDataset(df_val,   images_dir, get_transforms(False), is_train=False)
    test_dataset  = ChestXrayDataset(df_test,  images_dir, get_transforms(False), is_train=False)

    # ── Weighted sampler: oversample images with rare findings ─
    label_matrix = train_dataset.labels           # (N, 14)
    sample_weights = label_matrix.sum(axis=1) + 1.0   # at least weight=1 for "No Finding"
    sample_weights = sample_weights / sample_weights.sum()
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(train_dataset),
        replacement = True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    logger.info(
        "DataLoaders ready  |  train_batches=%d  val_batches=%d  test_batches=%d",
        len(train_loader), len(val_loader), len(test_loader)
    )

    return train_loader, val_loader, test_loader

# ============================================================
# FINDING IMAGE
# ============================================================

def find_image_path(image_name: str, base_dir: str) -> str:
    """
    NIH dataset mein images 12 subfolders mein hain:
    images_001/images/, images_002/images/ ... images_012/images/
    """
    # Pehle direct check
    direct = os.path.join(base_dir, image_name)
    if os.path.exists(direct):
        return direct

    # Subfolders check karo
    for i in range(1, 13):
        path = os.path.join(base_dir, f"images_{i:03d}", "images", image_name)
        if os.path.exists(path):
            return path

    return None   # image nahi mili

# ============================================================
# LOAD + PREPARE METADATA  (main entry point)
# ============================================================

def load_metadata(csv_path: str = LABELS_CSV) -> pd.DataFrame:
    """
    Load and validate NIH ChestX-ray14 Data_Entry_2017.csv.
    Returns clean DataFrame ready for splitting.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Labels CSV not found at: {csv_path}\n"
            "Download NIH ChestX-ray14 dataset from:\n"
            "  https://www.kaggle.com/datasets/nih-chest-xrays/data\n"
            "Place Data_Entry_2017.csv in the data/ directory."
        )

    logger.info("Loading metadata from: %s", csv_path)
    df = pd.read_csv(csv_path)
    df = validate_metadata(df)

    return df
