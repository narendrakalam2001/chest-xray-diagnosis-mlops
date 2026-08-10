# ============================================================
# PREDICTION SERVICE — Chest X-Ray Diagnosis ML System
# ============================================================

import io
import base64
import logging
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Optional

from src.config import PATHOLOGY_LABELS, CLINICAL_THRESHOLDS, HIGH_UNCERTAINTY_THRESHOLD
from src.preprocessing import build_eval_transform
from src.metrics import compute_prediction_entropy

logger = logging.getLogger(__name__)

# Critical pathologies requiring urgent radiologist review
CRITICAL_LABELS = {"Pneumonia", "Pneumothorax", "Mass", "Nodule", "Cardiomegaly"}


# ============================================================
# IMAGE → TENSOR
# ============================================================

def preprocess_image(
    image_bytes: bytes,
    device:      torch.device,
) -> torch.Tensor:
    """
    Load raw image bytes → preprocessed tensor ready for model.
    """
    img       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = build_eval_transform(use_clahe=True)
    tensor    = transform(img).unsqueeze(0)   # (1, C, H, W)
    return tensor.to(device)


def preprocess_image_base64(
    image_b64: str,
    device:    torch.device,
) -> torch.Tensor:
    """
    Decode base64 → preprocessed tensor.
    """
    img_bytes = base64.b64decode(image_b64)
    return preprocess_image(img_bytes, device)


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def run_inference(
    model:     nn.Module,
    tensor:    torch.Tensor,
) -> np.ndarray:
    """
    Run model forward pass → sigmoid probabilities.
    Returns (14,) array of probabilities.
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy().squeeze()   # (14,)
    return probs


# ============================================================
# STRUCTURE PREDICTION RESULT
# ============================================================

def build_prediction_result(
    probs:      np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """
    Convert probability array → structured clinical result.

    Returns:
        - findings:         list of {label, probability, severity}
        - normal:           True if no findings exceed threshold
        - critical_finding: True if any critical label is flagged
        - uncertainty_flag: True if entropy > threshold
        - mean_probability: average probability across all classes
        - full_probabilities: dict of all 14 label probabilities
    """
    probs      = np.clip(probs, 0, 1)
    y_pred     = (probs >= thresholds).astype(int)
    entropy_val= float(compute_prediction_entropy(probs.reshape(1, -1))[0])

    findings = []
    for i, lbl in enumerate(PATHOLOGY_LABELS):
        if y_pred[i] == 1:
            severity = _get_severity(lbl, float(probs[i]))
            findings.append({
                "label":         lbl,
                "probability":   round(float(probs[i]), 4),
                "threshold_used": round(float(thresholds[i]), 4),
                "severity":      severity,
                "critical":      lbl in CRITICAL_LABELS,
            })

    # Sort by probability descending
    findings = sorted(findings, key=lambda x: x["probability"], reverse=True)

    critical_finding = any(f["critical"] for f in findings)
    normal           = len(findings) == 0

    return {
        "findings":             findings,
        "n_findings":           len(findings),
        "normal":               normal,
        "critical_finding":     critical_finding,
        "uncertainty_flag":     entropy_val > HIGH_UNCERTAINTY_THRESHOLD,
        "prediction_entropy":   round(entropy_val, 4),
        "mean_probability":     round(float(probs.mean()), 4),
        "full_probabilities": {
            lbl: round(float(probs[i]), 4)
            for i, lbl in enumerate(PATHOLOGY_LABELS)
        },
        "clinical_recommendation": _get_recommendation(findings, critical_finding, entropy_val),
    }


def _get_severity(label: str, probability: float) -> str:
    """Map label + probability to severity tier."""
    if label in CRITICAL_LABELS:
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    else:
        return "LOW" if probability < 0.6 else "MODERATE"


def _get_recommendation(
    findings:         List[dict],
    critical_finding: bool,
    entropy:          float,
) -> str:
    """
    Clinical recommendation based on findings.
    Always defers to radiologist — AI provides decision SUPPORT only.
    """
    if not findings:
        return "No pathology detected — recommend routine follow-up per clinical protocol"

    if critical_finding:
        critical_names = [f["label"] for f in findings if f["critical"]]
        return (
            f"URGENT: Critical finding(s) detected — {', '.join(critical_names)}. "
            "Immediate radiologist review required."
        )

    if entropy > HIGH_UNCERTAINTY_THRESHOLD:
        return (
            "Uncertain prediction — model confidence is low. "
            "Radiologist review strongly recommended."
        )

    label_names = [f["label"] for f in findings]
    return (
        f"Finding(s) detected: {', '.join(label_names)}. "
        "Radiologist review recommended for confirmation."
    )


# ============================================================
# FULL PREDICTION ENTRY POINTS
# ============================================================

def predict_image(
    model:         nn.Module,
    image_bytes:   bytes,
    thresholds:    np.ndarray,
    device:        torch.device,
    return_gradcam: bool = False,
) -> dict:
    """
    Full prediction pipeline for a raw image bytes input.
    """
    tensor = preprocess_image(image_bytes, device)
    probs  = run_inference(model, tensor)
    result = build_prediction_result(probs, thresholds)

    if return_gradcam:
        result["gradcam"] = _generate_gradcam_b64(model, tensor, probs, thresholds, device)

    logger.info(
        "Prediction  |  n_findings=%d  critical=%s  entropy=%.4f",
        result["n_findings"],
        result["critical_finding"],
        result["prediction_entropy"],
    )
    return result


def predict_image_base64(
    model:         nn.Module,
    image_b64:     str,
    thresholds:    np.ndarray,
    device:        torch.device,
    return_gradcam: bool = False,
) -> dict:
    """
    Full prediction pipeline for a base64-encoded image.
    """
    tensor = preprocess_image_base64(image_b64, device)
    probs  = run_inference(model, tensor)
    result = build_prediction_result(probs, thresholds)

    if return_gradcam:
        result["gradcam"] = _generate_gradcam_b64(model, tensor, probs, thresholds, device)

    return result


def _generate_gradcam_b64(
    model:     nn.Module,
    tensor:    torch.Tensor,
    probs:     np.ndarray,
    thresholds: np.ndarray,
    device:    torch.device,
) -> dict:
    """
    Generate Grad-CAM heatmaps for top detected classes.
    Returns dict of {label: base64_png} for each detected finding.
    """
    from src.config import GRADCAM_TARGET_LAYER
    from src.evaluation import GradCAM
    import matplotlib.pyplot as plt
    import io, base64

    gradcam = GradCAM(model, GRADCAM_TARGET_LAYER)
    result  = {}

    detected = np.where(probs >= thresholds)[0][:3]   # top 3 findings max

    for i in detected:
        lbl     = PATHOLOGY_LABELS[i]
        heatmap = gradcam.generate(tensor, int(i), device)

        # Encode heatmap as base64 PNG
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(heatmap, cmap="jet")
        ax.axis("off")
        ax.set_title(f"{lbl}: {probs[i]:.3f}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        plt.close()
        buf.seek(0)
        result[lbl] = base64.b64encode(buf.read()).decode()

    return result
