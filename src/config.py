# ============================================================
# CONFIGURATION — Chest X-Ray Diagnosis ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Dataset ──────────────────────────────────────────────────
# ── Training — 2GB VRAM ke liye optimized ────────────────────
IMAGE_SIZE   = (112, 112)    # 224 → 112  (VRAM 4x kam lagegi)
BATCH_SIZE   = 8             # 32 → 8     (VRAM fit hogi)
NUM_WORKERS  = 2             # 4 → 2      (CPU load kam)

# ── NIH ChestX-ray14 — 14 pathology labels ───────────────────
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

NUM_CLASSES    = len(PATHOLOGY_LABELS)   # 14

# ── Training ─────────────────────────────────────────────────
EPOCHS              = 30
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-4
LR_PATIENCE         = 5        # ReduceLROnPlateau patience
EARLY_STOP_PATIENCE = 8        # EarlyStopping patience
GRAD_CLIP_VALUE     = 1.0      # gradient clipping for stability

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS            = 5

# ── Class imbalance handling ──────────────────────────────────
# NIH dataset is heavily imbalanced — use weighted BCE
USE_WEIGHTED_LOSS   = True
POS_WEIGHT_SCALE    = 2.0      # multiplier on computed pos_weights

# ── Thresholds (per-label, tunable) ──────────────────────────
# Default; will be tuned per class during evaluation
DEFAULT_THRESHOLD   = 0.5

# Clinical thresholds — higher sensitivity for critical conditions
CLINICAL_THRESHOLDS = {
    "Pneumonia":       0.35,   # high sensitivity — miss = patient harm
    "Pneumothorax":    0.30,   # high sensitivity — life-threatening
    "Cardiomegaly":    0.40,
    "Mass":            0.40,
    "Nodule":          0.45,
    "default":         0.50,
}

# ── PSI drift thresholds ─────────────────────────────────────
PSI_MODERATE        = 0.10
PSI_HIGH            = 0.20

# ── Monitoring ───────────────────────────────────────────────
MEAN_AUC_ALERT      = 0.75    # alert if mean AUC drops below this
HIGH_UNCERTAINTY_THRESHOLD = 0.15  # flag if entropy > threshold

# ── Champion-Challenger gates ─────────────────────────────────
MIN_MEAN_AUC_IMPROVEMENT    = 0.005   # challenger must beat by 0.5% mean AUC
MIN_MEAN_AUC_THRESHOLD      = 0.75    # challenger must clear this bar
MAX_GENERALIZATION_GAP      = 0.08    # max train-val AUC gap

# ── Backbone options ──────────────────────────────────────────
BACKBONE_OPTIONS    = ["densenet121", "resnet50", "efficientnet_b0"]
DEFAULT_BACKBONE    = "densenet121"   # DenseNet121 — NIH paper baseline

# ── Grad-CAM ─────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "features.denseblock4.denselayer16.conv2"  # DenseNet121

# ── Paths ─────────────────────────────────────────────────────
# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR   = "xray_models"
METRICS_LOG = "xray_models/metrics_log.csv"

# ↓ SIRF YE 3 LINES CHANGE KARO
DATA_DIR   = r"D:\Data Science Datasets\NIH Chest X-rays"
IMAGES_DIR = r"D:\Data Science Datasets\NIH Chest X-rays"
LABELS_CSV = r"D:\Data Science Datasets\NIH Chest X-rays\Data_Entry_2017.csv"

# ↓ YE 3 LINES BILKUL MAT CHHUO — SAME RAKHO
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)