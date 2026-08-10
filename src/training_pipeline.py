# ============================================================
# TRAINING PIPELINE — Chest X-Ray Diagnosis ML System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt

from src.config import (
    RANDOM_STATE, EPOCHS, MODEL_DIR, DEFAULT_BACKBONE, BACKBONE_OPTIONS,
    GRAD_CLIP_VALUE, PATHOLOGY_LABELS, CLINICAL_THRESHOLDS
)
from src.data_loader    import (load_metadata, patient_level_split,
                                 create_dataloaders, compute_pos_weights)
from src.preprocessing  import build_threshold_array
from src.model_tuning   import (build_model, build_loss, build_optimizer,
                                  build_scheduler, EarlyStopping, BACKBONE_GRID)
from src.evaluation     import (evaluate_model, plot_roc_curves, plot_auc_barchart,
                                  save_model_and_card, mlflow_log_run, visualize_gradcam,
                                  GradCAM)
from src.leakage_check  import detect_all_leakage
from src.metrics        import (compute_per_class_auc, compute_mean_auc,
                                  tune_thresholds_per_class, psi_per_class_score)
from src.model_loader   import run_challenger_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# SINGLE EPOCH TRAINING
# ============================================================

def train_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    GradScaler,
) -> float:
    """
    Train one epoch with mixed-precision (AMP) and gradient clipping.
    Returns average train loss.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.float().to(device)   # ← .float() add karo
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=False):  # 2GB VRAM — AMP off
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.debug("Train batch %d/%d  loss=%.4f",
                         batch_idx, len(loader), loss.item())

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

def validate_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple:
    """
    Validate one epoch.
    Returns: (val_loss, mean_auc, per_class_auc_dict)
    """
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.float().to(device)   # ← .float() add karo
            labels = labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    y_prob  = np.vstack(all_probs)
    y_true  = np.vstack(all_labels)

    auc_dict = compute_per_class_auc(y_true, y_prob)
    mean_auc = compute_mean_auc(auc_dict)

    return total_loss / len(loader), mean_auc, auc_dict


# ============================================================
# TRAIN SINGLE BACKBONE
# ============================================================

def train_backbone(
    backbone:     str,
    df_train:     pd.DataFrame,
    df_val:       pd.DataFrame,
    images_dir:   str,
    pos_weights:  torch.Tensor,
    device:       torch.device,
) -> tuple:
    """
    Train a single backbone variant.
    Returns: (model, train_auc, val_auc, history)
    """
    logger.info("=" * 60)
    logger.info("Training backbone: %s", backbone)
    logger.info("=" * 60)

    from src.data_loader import ChestXrayDataset, get_transforms
    from torch.utils.data import DataLoader

    # ── Dataloaders ───────────────────────────────────────────
    cfg   = BACKBONE_GRID.get(backbone, BACKBONE_GRID["DenseNet121"])
    train_ds = ChestXrayDataset(df_train, images_dir, get_transforms(True),  is_train=True)
    val_ds   = ChestXrayDataset(df_val,   images_dir, get_transforms(False), is_train=False)

    from src.config import BATCH_SIZE, NUM_WORKERS
    import torch
    from torch.utils.data import WeightedRandomSampler

    label_matrix   = train_ds.labels
    sample_weights = label_matrix.sum(axis=1) + 1.0
    sample_weights /= sample_weights.sum()
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.float32),
        len(train_ds), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────
    model     = build_model(backbone=cfg["backbone"], pretrained=True, unfreeze_n=cfg["unfreeze_n"])
    model     = model.to(device)
    criterion = build_loss(pos_weights=pos_weights.to(device))
    optimizer = build_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, scheduler=cfg["scheduler"])
    stopper   = EarlyStopping(
        checkpoint=os.path.join(MODEL_DIR, f"best_{backbone}.pt")
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history   = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc   = 0.0
    best_train_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_auc, _ = validate_epoch(model, val_loader, criterion, device)

        # Quick train AUC on a sample of train loader
        _, train_auc_cur, _ = validate_epoch(model, train_loader, criterion, device)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_auc"].append(round(val_auc, 4))

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  [%.0fs]",
            epoch, EPOCHS, train_loss, val_loss, val_auc, elapsed
        )

        # LR scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_auc)
        else:
            scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_train_auc = train_auc_cur

        if stopper.step(model, val_auc):
            break

    # Load best checkpoint
    if os.path.exists(stopper.checkpoint):
        checkpoint = torch.load(stopper.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    logger.info(
        "Backbone %s done  |  best_val_auc=%.4f  best_train_auc=%.4f",
        backbone, best_val_auc, best_train_auc
    )

    return model, best_train_auc, best_val_auc, history


# ============================================================
# TRAINING CURVE PLOT
# ============================================================

def plot_training_history(history: dict, backbone: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title(f"{backbone} — Loss")
    axes[0].legend()

    axes[1].plot(history["val_auc"],    label="Val Mean AUC", color="green")
    axes[1].axhline(0.80, color="orange", linestyle="--", label="0.80 target")
    axes[1].set_title(f"{backbone} — Validation AUC")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("docs/plots", exist_ok=True)
    path = f"docs/plots/training_history_{backbone}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    logger.info("Training history plot saved: %s", path)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training(
    backbone:    str       = DEFAULT_BACKBONE,
    images_dir:  str       = None,
    labels_csv:  str       = None,
):
    """
    Full end-to-end training pipeline.

    Steps:
      1. Load & validate metadata
      2. Patient-level split (leakage-safe)
      3. Leakage detection
      4. Compute class weights
      5. Train backbone(s)
      6. Full evaluation on test set
      7. Grad-CAM visualization
      8. Save model + model card
      9. Champion-Challenger comparison
      10. MLflow logging
    """
    from src.config import IMAGES_DIR, LABELS_CSV
    images_dir = images_dir or IMAGES_DIR
    labels_csv = labels_csv or LABELS_CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ──────────────────────────────────────────────────────────
    # 1. LOAD & VALIDATE METADATA
    # ──────────────────────────────────────────────────────────
    logger.info("Loading metadata from: %s", labels_csv)
    df = load_metadata(labels_csv)

    # ──────────────────────────────────────────────────────────
    # 2. PATIENT-LEVEL SPLIT
    # ──────────────────────────────────────────────────────────
    df_train, df_val, df_test = patient_level_split(df)

    # ──────────────────────────────────────────────────────────
    # 3. LEAKAGE DETECTION
    # ──────────────────────────────────────────────────────────
    leak_warnings = detect_all_leakage(df_train, df_val, df_test)
    if leak_warnings:
        for w in leak_warnings:
            logger.error("LEAKAGE: %s", w)
        raise RuntimeError("Training aborted — leakage detected. Fix data splits first.")
    else:
        logger.info("No leakage detected — safe to proceed")

    # ──────────────────────────────────────────────────────────
    # 4. CLASS WEIGHTS
    # ──────────────────────────────────────────────────────────
    from src.config import POS_WEIGHT_SCALE
    pos_weights = compute_pos_weights(df_train, scale=POS_WEIGHT_SCALE)
    logger.info("Pos weights range: [%.2f, %.2f]",
                pos_weights.min().item(), pos_weights.max().item())

    # ──────────────────────────────────────────────────────────
    # 5. TRAIN BACKBONE
    # ──────────────────────────────────────────────────────────
    model, train_auc, val_auc, history = train_backbone(
        backbone, df_train, df_val, images_dir, pos_weights, device
    )

    # ──────────────────────────────────────────────────────────
    # 6. FULL TEST EVALUATION
    # ──────────────────────────────────────────────────────────
    from src.data_loader import ChestXrayDataset, get_transforms
    from torch.utils.data import DataLoader
    from src.config import BATCH_SIZE, NUM_WORKERS

    test_ds     = ChestXrayDataset(df_test, images_dir, get_transforms(False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    summary, y_true, y_prob, y_pred, cls_report, clinical_df = evaluate_model(
        model, test_loader, device, split_name="test"
    )
    summary["train_auc"] = round(train_auc, 4)

    # ──────────────────────────────────────────────────────────
    # 7. PLOTS
    # ──────────────────────────────────────────────────────────
    plot_training_history(history, backbone)
    plot_roc_curves(y_true, y_prob)
    plot_auc_barchart(summary["per_class_auc"])

    # ──────────────────────────────────────────────────────────
    # 8. SAVE MODEL + MODEL CARD
    # ──────────────────────────────────────────────────────────
    import numpy as np
    thresholds  = tune_thresholds_per_class(y_true, y_prob, clinical_labels=CLINICAL_THRESHOLDS)
    model_path, card_path = save_model_and_card(
        model, backbone, backbone, summary, thresholds, cls_report
    )

    # Save optimal thresholds to registry
    import numpy as np
    np.save(os.path.join(MODEL_DIR, "optimal_thresholds.npy"), thresholds)

    # Save PSI reference distribution
    np.save(os.path.join(MODEL_DIR, "ref_probs.npy"), y_prob)

    # Save classification report
    cls_report.to_csv(os.path.join(MODEL_DIR, "test_classification_report.csv"), index=False)

    # ──────────────────────────────────────────────────────────
    # 9. CHAMPION-CHALLENGER
    # ──────────────────────────────────────────────────────────
    challenger_result = run_challenger_comparison(
        challenger_name        = backbone,
        challenger_mean_auc    = summary["mean_auc"],
        challenger_val_auc     = val_auc,
        challenger_train_auc   = train_auc,
        challenger_model_path  = model_path,
        challenger_card_path   = card_path,
    )

    print("\n" + "=" * 60)
    print("CHAMPION-CHALLENGER RESULT")
    print(f"  Decision   : {challenger_result['decision']}")
    print(f"  Reason     : {challenger_result['reason']}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 10. MLFLOW
    # ──────────────────────────────────────────────────────────
    mlflow_log_run(backbone, backbone, summary, model_path)

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Backbone   : {backbone}")
    print(f"  Mean AUC   : {summary['mean_auc']:.4f}")
    print(f"  Mean F1    : {summary['mean_f1']:.4f}")
    print(f"  Mean AP    : {summary['mean_ap']:.4f}")
    print(f"  Model path : {model_path}")
    print("=" * 60)

    return model, summary, thresholds


if __name__ == "__main__":
    run_training()
