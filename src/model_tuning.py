# ============================================================
# MODEL TUNING — Chest X-Ray Diagnosis ML System
# ============================================================
# Multi-label classification with pretrained CNN backbones.
# Backbones:
#   - DenseNet121  (CheXNet baseline — Rajpurkar et al. 2017)
#   - ResNet50     (strong general-purpose baseline)
#   - EfficientNet-B0 (efficient, strong performance)
# All backbones: ImageNet pretrained, final FC replaced with
#   14-output sigmoid head (multi-label).
# ============================================================

import logging
import numpy as np
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import models

from src.config import (
    NUM_CLASSES, PATHOLOGY_LABELS, LEARNING_RATE, WEIGHT_DECAY,
    EPOCHS, EARLY_STOP_PATIENCE, LR_PATIENCE, GRAD_CLIP_VALUE,
    DEFAULT_BACKBONE, BACKBONE_OPTIONS, RANDOM_STATE
)

logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_STATE)


# ============================================================
# MULTI-LABEL CLASSIFICATION HEAD
# ============================================================

class MultiLabelHead(nn.Module):
    """
    Classification head for multi-label X-ray diagnosis.
    Replaces final FC layer of any backbone.

    Architecture:
        GlobalAvgPool → Dropout(0.3) → FC(hidden) → ReLU →
        Dropout(0.2) → FC(14) → Sigmoid

    No sigmoid at training time (BCEWithLogitsLoss is numerically
    stable); sigmoid applied during inference.
    """

    def __init__(
        self,
        in_features:  int,
        num_classes:  int   = NUM_CLASSES,
        hidden_dim:   int   = 512,
        dropout1:     float = 0.3,
        dropout2:     float = 0.2,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ============================================================
# BACKBONE FACTORY
# ============================================================

def build_model(
    backbone:    str  = DEFAULT_BACKBONE,
    pretrained:  bool = True,
    freeze_base: bool = True,
    unfreeze_n:  int  = 2,          # unfreeze last N blocks for fine-tuning
) -> nn.Module:
    """
    Build multi-label classification model from pretrained backbone.

    Freezing strategy:
        freeze_base=True  → freeze all backbone layers first
        unfreeze_n        → then unfreeze last N blocks for fine-tuning
        This is standard practice: first train head, then fine-tune end.

    Args:
        backbone:    one of 'densenet121', 'resnet50', 'efficientnet_b0'
        pretrained:  use ImageNet pretrained weights
        freeze_base: freeze backbone initially
        unfreeze_n:  how many terminal backbone blocks to unfreeze

    Returns:
        nn.Module with 14-class output
    """

    backbone = backbone.lower()

    if backbone == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier.in_features
        model.classifier = MultiLabelHead(in_features)

        if freeze_base:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            # Unfreeze last N denseblocks
            blocks_to_unfreeze = [f"denseblock{4 - i}" for i in range(unfreeze_n)]
            for name, param in model.named_parameters():
                if any(blk in name for blk in blocks_to_unfreeze):
                    param.requires_grad = True

    elif backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.fc.in_features
        model.fc = MultiLabelHead(in_features)

        if freeze_base:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
            # Unfreeze last N layers (layer4, layer3, etc.)
            layers_to_unfreeze = [f"layer{4 - i}" for i in range(unfreeze_n)]
            for name, param in model.named_parameters():
                if any(lyr in name for lyr in layers_to_unfreeze):
                    param.requires_grad = True

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = model.classifier[1].in_features
        model.classifier = MultiLabelHead(in_features)

        if freeze_base:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            # Unfreeze last N MBConv blocks
            for i, block in enumerate(reversed(list(model.features))):
                if i >= unfreeze_n:
                    break
                for param in block.parameters():
                    param.requires_grad = True

    else:
        raise ValueError(
            f"Unknown backbone: '{backbone}'. "
            f"Choose from: {BACKBONE_OPTIONS}"
        )

    # ── Log trainable params ──────────────────────────────────
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %-20s  total_params=%s  trainable=%s  (%.1f%%)",
        backbone, f"{total:,}", f"{trainable:,}", 100 * trainable / total
    )

    return model


# ============================================================
# LOSS FUNCTION
# ============================================================

def build_loss(pos_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    BCEWithLogitsLoss with per-class positive weights.
    Handles severe class imbalance in NIH dataset.

    BCEWithLogitsLoss is numerically stable vs BCE(sigmoid(logits)).
    pos_weight[i] = neg_count_i / pos_count_i (from compute_pos_weights)
    """
    if pos_weights is not None:
        logger.info("Using weighted BCEWithLogitsLoss  |  pos_weight range: [%.2f, %.2f]",
                    pos_weights.min().item(), pos_weights.max().item())
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        logger.info("Using unweighted BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================

def build_optimizer(
    model:          nn.Module,
    lr:             float = LEARNING_RATE,
    weight_decay:   float = WEIGHT_DECAY,
    use_layer_lr:   bool  = True,
) -> optim.Optimizer:
    """
    AdamW optimizer with layer-wise learning rates.
    Head gets full LR; backbone gets reduced LR (0.1x).
    Standard practice for fine-tuning pretrained medical models.
    """
    if use_layer_lr:
        head_params     = []
        backbone_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in ["classifier", "fc", "head"]):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": head_params,     "lr": lr,         "name": "head"},
            {"params": backbone_params, "lr": lr * 0.1,   "name": "backbone"},
        ]
        logger.info("Layer-wise LR  |  head_lr=%.2e  backbone_lr=%.2e", lr, lr * 0.1)
    else:
        param_groups = [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": lr}]

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer:    optim.Optimizer,
    scheduler:    str   = "plateau",
    epochs:       int   = EPOCHS,
) -> object:
    """
    LR scheduler options:
        'plateau' — ReduceLROnPlateau (val AUC-based)
        'cosine'  — CosineAnnealingLR (smooth decay)
    """
    if scheduler == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        logger.info("Scheduler: CosineAnnealingLR  T_max=%d", epochs)
    else:
        sched = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE, min_lr=1e-7
    )
        logger.info("Scheduler: ReduceLROnPlateau  patience=%d", LR_PATIENCE)

    return sched


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Stop training when val_auc stops improving.
    Saves best model checkpoint.
    """

    def __init__(
        self,
        patience:    int   = EARLY_STOP_PATIENCE,
        min_delta:   float = 0.001,
        checkpoint:  str   = "xray_models/best_checkpoint.pt",
    ):
        self.patience    = patience
        self.min_delta   = min_delta
        self.checkpoint  = checkpoint
        self.best_score  = -np.inf
        self.counter     = 0
        self.stopped     = False

    def step(self, model: nn.Module, score: float) -> bool:
        """
        Returns True if training should stop.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            torch.save(model.state_dict(), self.checkpoint)
            logger.info("EarlyStopping: new best  val_auc=%.4f  saved to %s",
                        score, self.checkpoint)
            return False
        else:
            self.counter += 1
            logger.info("EarlyStopping: no improvement  counter=%d/%d",
                        self.counter, self.patience)
            if self.counter >= self.patience:
                self.stopped = True
                logger.info("EarlyStopping triggered at patience=%d", self.patience)
                return True
            return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best checkpoint back into model."""
        if os.path.exists(self.checkpoint):
            model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
            logger.info("Loaded best checkpoint from %s", self.checkpoint)
        return model


# ============================================================
# HYPERPARAMETER GRID  (for backbone comparison)
# ============================================================

BACKBONE_GRID = {
    "DenseNet121": {
        "backbone":      "densenet121",
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "dropout1":      0.3,
        "dropout2":      0.2,
        "hidden_dim":    512,
        "unfreeze_n":    2,
        "scheduler":     "plateau",
    },
    "ResNet50": {
        "backbone":      "resnet50",
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "dropout1":      0.4,
        "dropout2":      0.2,
        "hidden_dim":    512,
        "unfreeze_n":    1,
        "scheduler":     "plateau",
    },
    "EfficientNet_B0": {
        "backbone":      "efficientnet_b0",
        "lr":            2e-4,
        "weight_decay":  1e-5,
        "dropout1":      0.3,
        "dropout2":      0.2,
        "hidden_dim":    256,
        "unfreeze_n":    2,
        "scheduler":     "cosine",
    },
}


import os  # noqa: E402 — placed here to avoid top-level dep
