# ============================================================
# EVALUATION — Chest X-Ray Diagnosis ML System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn as nn
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import roc_curve, auc, RocCurveDisplay

from src.config import (
    PATHOLOGY_LABELS, NUM_CLASSES, MODEL_DIR, RANDOM_STATE,
    CLINICAL_THRESHOLDS, GRADCAM_TARGET_LAYER
)
from src.metrics import (
    compute_per_class_auc, compute_mean_auc, compute_classification_report,
    compute_clinical_metrics, clinical_cost_evaluation, compare_to_radiologist,
    tune_thresholds_per_class, flag_uncertain_predictions
)
from src.preprocessing import denormalize_tensor, build_threshold_array

logger = logging.getLogger(__name__)


# ============================================================
# INFERENCE — get probabilities from DataLoader
# ============================================================

def predict_dataloader(
    model:      nn.Module,
    loader,
    device:     torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run inference on a DataLoader.

    Returns:
        y_true:      (N, 14) ground truth binary labels
        y_prob:      (N, 14) predicted probabilities
        image_names: list of image filenames
    """
    model.eval()
    model.to(device)

    all_probs   = []
    all_labels  = []
    all_names   = []

    with torch.no_grad():
        for batch in loader:
            images, labels, names = batch
            images = images.to(device)

            logits = model(images)
            probs  = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_names.extend(list(names))

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    logger.info("Inference complete  |  n=%d  y_true shape=%s", len(y_true), y_true.shape)
    return y_true, y_prob, all_names


# ============================================================
# FULL EVALUATION SUITE
# ============================================================

def evaluate_model(
    model:         nn.Module,
    test_loader,
    device:        torch.device,
    split_name:    str = "test",
) -> Dict:
    """
    Full evaluation: AUC, F1, clinical metrics, cost evaluation.
    Returns dict of all metrics.
    """
    logger.info("Evaluating model on %s set ...", split_name)
    y_true, y_prob, image_names = predict_dataloader(model, test_loader, device)

    # ── Per-class AUC ─────────────────────────────────────────
    auc_dict   = compute_per_class_auc(y_true, y_prob)
    mean_auc   = compute_mean_auc(auc_dict)

    # ── Threshold tuning ──────────────────────────────────────
    clinical_thr_array = build_threshold_array(CLINICAL_THRESHOLDS)
    opt_thresholds     = tune_thresholds_per_class(
        y_true, y_prob, clinical_labels=CLINICAL_THRESHOLDS
    )
    y_pred = (y_prob >= opt_thresholds).astype(int)

    # ── Classification report ─────────────────────────────────
    cls_report    = compute_classification_report(y_true, y_pred, y_prob)
    cls_report    = compare_to_radiologist(cls_report)

    # ── Clinical metrics ──────────────────────────────────────
    clinical_df   = compute_clinical_metrics(y_true, y_pred)

    # ── Cost evaluation ───────────────────────────────────────
    cost_result   = clinical_cost_evaluation(y_true, y_pred)

    # ── Uncertainty ───────────────────────────────────────────
    uncertain_mask = flag_uncertain_predictions(y_prob)
    uncertain_pct  = float(uncertain_mask.mean())

    # ── Summary ───────────────────────────────────────────────
    mean_f1      = float(cls_report["f1"].dropna().mean())
    mean_ap      = float(cls_report["avg_precision"].dropna().mean())

    summary = {
        "split":               split_name,
        "mean_auc":            round(mean_auc, 4),
        "mean_f1":             round(mean_f1, 4),
        "mean_ap":             round(mean_ap, 4),
        "uncertain_fraction":  round(uncertain_pct, 4),
        "total_samples":       len(y_true),
        "per_class_auc":       {k: round(v, 4) for k, v in auc_dict.items() if not np.isnan(v)},
        "clinical_cost":       cost_result,
        "optimal_thresholds":  opt_thresholds.tolist(),
    }

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS — %s", split_name.upper())
    logger.info("  Mean AUC : %.4f", mean_auc)
    logger.info("  Mean F1  : %.4f", mean_f1)
    logger.info("  Mean AP  : %.4f", mean_ap)
    logger.info("  Uncertain: %.1f%%", uncertain_pct * 100)
    logger.info("=" * 60)

    print("\nPer-class AUC:")
    for lbl, auc_val in sorted(auc_dict.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True):
        if not np.isnan(auc_val):
            print(f"  {lbl:<22} AUC={auc_val:.4f}")

    print("\nClassification Report:")
    print(cls_report[["label", "n_positive", "f1", "roc_auc", "vs_radiologist"]].to_string(index=False))

    return summary, y_true, y_prob, y_pred, cls_report, clinical_df


# ============================================================
# ROC CURVES PLOT
# ============================================================

def plot_roc_curves(
    y_true:   np.ndarray,
    y_prob:   np.ndarray,
    save_path: str = "docs/plots/roc_curves.png",
) -> None:
    """
    Plot per-class ROC curves on a single figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, lbl in enumerate(PATHOLOGY_LABELS):
        ax   = axes[i]
        y_t  = y_true[:, i]
        y_p  = y_prob[:, i]

        if y_t.sum() == 0:
            ax.set_title(f"{lbl} (no positives)")
            continue

        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_val     = auc(fpr, tpr)

        ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"AUC={auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(lbl)
        ax.legend(loc="lower right", fontsize=8)

    # Hide unused subplots
    for j in range(len(PATHOLOGY_LABELS), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Per-Class ROC Curves — Chest X-Ray Diagnosis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("ROC curves saved to %s", save_path)


# ============================================================
# GRAD-CAM — visual explainability
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.
    Produces heatmap highlighting image regions that activated
    a specific pathology prediction — used for clinical explainability.

    Reference: Selvaraju et al. 2017, "Grad-CAM: Visual Explanations
    from Deep Networks via Gradient-based Localization"
    """

    def __init__(self, model: nn.Module, target_layer: str):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks(target_layer)

    def _find_layer(self, target_layer: str) -> nn.Module:
        """Traverse model to find layer by dot-path name."""
        parts = target_layer.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Try integer index for Sequential
                try:
                    module = module[int(part)]
                except Exception:
                    logger.warning("Grad-CAM: layer '%s' not found", target_layer)
                    return None
        return module

    def _register_hooks(self, target_layer: str):
        layer = self._find_layer(target_layer)
        if layer is None:
            logger.warning("Grad-CAM hooks not registered — layer not found")
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx:    int,
        device:       torch.device,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image and class.

        Args:
            image_tensor: (1, C, H, W) preprocessed image
            class_idx:    pathology class index (0-13)
            device:       CPU or CUDA

        Returns:
            (H, W) heatmap array, values in [0, 1]
        """
        self.model.eval()
        self.model.to(device)

        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)
        score  = logits[0, class_idx]

        # Backward pass for this class
        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM: gradients/activations not captured")
            return np.zeros((224, 224))

        # Grad-CAM formula: pool gradients → weight feature maps → ReLU
        weights    = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam        = (weights * self.activations).sum(dim=1)         # (1, H, W)
        cam        = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-9:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Upsample to image size
        from PIL import Image as PILImage
        cam_img = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((224, 224), PILImage.BILINEAR)
        return np.array(cam_img) / 255.0


def visualize_gradcam(
    image_tensor:  torch.Tensor,
    heatmap:       np.ndarray,
    label:         str,
    probability:   float,
    save_path:     str = "docs/plots/gradcam_example.png",
) -> None:
    """
    Overlay Grad-CAM heatmap on original X-ray image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Denormalize image
    img_tensor = denormalize_tensor(image_tensor.squeeze())
    img_np     = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_gray   = img_np.mean(axis=2)   # grayscale for X-ray display

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM: {label}")
    axes[1].axis("off")

    axes[2].imshow(img_gray, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title(f"Overlay  |  P={probability:.3f}")
    axes[2].axis("off")

    plt.suptitle(f"Grad-CAM Explainability — {label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Grad-CAM visualization saved to %s", save_path)


# ============================================================
# AUC BAR CHART
# ============================================================

def plot_auc_barchart(
    auc_dict:  Dict[str, float],
    save_path: str = "docs/plots/auc_barchart.png",
) -> None:
    """
    Bar chart of per-class AUC with radiologist benchmark line.
    """
    from src.metrics import RADIOLOGIST_BENCHMARKS
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = list(auc_dict.keys())
    values = list(auc_dict.values())
    bench  = [RADIOLOGIST_BENCHMARKS.get(lbl, 0.75) for lbl in labels]

    x     = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(x, values, color="steelblue", alpha=0.8, label="Model AUC")
    ax.scatter(x, bench, color="red", zorder=5, marker="D", s=60, label="Radiologist F1 (approx)")

    ax.axhline(0.8, color="orange", linestyle="--", linewidth=1, label="AUC=0.8 target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim([0, 1.05])
    ax.set_title("Per-Class AUC vs Radiologist Benchmark")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("AUC bar chart saved to %s", save_path)


# ============================================================
# SAVE MODEL + METADATA
# ============================================================

def save_model_and_card(
    model:           nn.Module,
    model_name:      str,
    backbone:        str,
    metrics_summary: Dict,
    thresholds:      np.ndarray,
    cls_report:      pd.DataFrame,
) -> Tuple[str, str]:
    """
    Save model weights (.pt) + model card JSON.
    Returns: (model_path, card_path)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version   = f"{model_name}_v{timestamp}"

    # ── Save model weights ────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"xray_model_{version}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone":         backbone,
        "thresholds":       thresholds.tolist(),
        "model_name":       version,
        "timestamp":        timestamp,
    }, model_path)
    logger.info("Model saved: %s", model_path)

    # ── Build model card ──────────────────────────────────────
    auc_dict = metrics_summary.get("per_class_auc", {})
    card = {
        "model_name":       version,
        "backbone":         backbone,
        "task":             "Multi-label Chest X-Ray Pathology Detection",
        "dataset":          "NIH ChestX-ray14 (112,120 images, 14 classes)",
        "num_classes":      NUM_CLASSES,
        "pathology_labels": PATHOLOGY_LABELS,
        "trained_at":       datetime.now().isoformat(),
        "metrics": {
            "mean_auc":    metrics_summary.get("mean_auc",  0),
            "mean_f1":     metrics_summary.get("mean_f1",   0),
            "mean_ap":     metrics_summary.get("mean_ap",   0),
            "per_class_auc": auc_dict,
        },
        "thresholds": {
            "per_class": {
                lbl: round(float(thresholds[i]), 4)
                for i, lbl in enumerate(PATHOLOGY_LABELS)
            },
            "tuning_strategy": "F1-optimal with clinical overrides for critical labels",
        },
        "pipeline_config": {
            "image_size":    [224, 224],
            "backbone":      backbone,
            "pretrained":    "ImageNet",
            "augmentation":  "CLAHE + HFlip + SafeRotation + ColorJitter",
            "loss":          "BCEWithLogitsLoss with pos_weights",
        },
        "clinical_notes": {
            "pneumonia_threshold":    CLINICAL_THRESHOLDS.get("Pneumonia"),
            "pneumothorax_threshold": CLINICAL_THRESHOLDS.get("Pneumothorax"),
            "intended_use":           "Decision support for radiologists — NOT for autonomous diagnosis",
            "limitations":            "Trained on NIH labels derived from NLP — may contain label noise",
        },
        "business_impact": {
            "use_case":       "Radiology AI triage — Apollo, Fortis, Max Healthcare, Niramai",
            "speed_benefit":  "Prioritize high-risk cases in radiology queue",
            "cost_saving":    "Reduce radiologist review time by 30-40% on normal cases",
        },
        "google_model_card_version": "1.0",
    }

    card_path = os.path.join(MODEL_DIR, f"model_card_{version}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    logger.info("Model card saved: %s", card_path)

    return model_path, card_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    model_name:      str,
    backbone:        str,
    metrics_summary: Dict,
    model_path:      str,
) -> None:
    """Log training run to MLflow."""
    try:
        import mlflow
        mlflow.set_experiment("chest-xray-diagnosis")
        mlflow.set_tracking_uri("file:./mlruns")

        with mlflow.start_run(run_name=model_name):
            # Log params
            mlflow.log_param("backbone",    backbone)
            mlflow.log_param("num_classes", NUM_CLASSES)

            # Log metrics
            mlflow.log_metric("mean_auc",  metrics_summary.get("mean_auc", 0))
            mlflow.log_metric("mean_f1",   metrics_summary.get("mean_f1",  0))
            mlflow.log_metric("mean_ap",   metrics_summary.get("mean_ap",  0))

            per_class = metrics_summary.get("per_class_auc", {})
            for lbl, val in per_class.items():
                safe_lbl = lbl.replace(" ", "_").lower()
                mlflow.log_metric(f"auc_{safe_lbl}", val)

            # Log model artifact
            mlflow.log_artifact(model_path)

        logger.info("MLflow run logged: %s", model_name)

    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
