# ============================================================
# MODEL LOADER + CHAMPION-CHALLENGER — Chest X-Ray Diagnosis
# ============================================================
#
# CHALLENGER MODEL SYSTEM:
#   Real clinical AI systems never blindly deploy a new model.
#   They run a formal Champion vs Challenger evaluation:
#
#   Champion  = current production model (latest_model.json)
#   Challenger = newly trained model (from training_pipeline)
#
#   Promotion logic:
#     Challenger promoted ONLY if it beats Champion on ALL 3 gates:
#       1. Mean AUC improvement  >= MIN_MEAN_AUC_IMPROVEMENT (0.5%)
#       2. Mean AUC absolute     >= MIN_MEAN_AUC_THRESHOLD   (0.75)
#       3. Train-val AUC gap     <= MAX_GENERALIZATION_GAP   (8%)
#
#   If challenger loses → champion stays, challenger archived.
#   Full comparison saved to xray_models/challenger_log.json
# ============================================================

import os
import json
import torch
import logging
import time

from src.config import (
    MODEL_DIR, NUM_CLASSES, PATHOLOGY_LABELS,
    MIN_MEAN_AUC_IMPROVEMENT, MIN_MEAN_AUC_THRESHOLD, MAX_GENERALIZATION_GAP
)

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model(device=None):
    """
    Reads xray_models/latest_model.json → loads .pt weights.
    Returns: (model, thresholds, backbone)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    backbone   = checkpoint.get("backbone", "densenet121")
    thresholds = checkpoint.get("thresholds", [0.5] * NUM_CLASSES)

    # Rebuild model architecture
    from src.model_tuning import build_model
    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    import numpy as np
    thresholds_np = np.array(thresholds, dtype=np.float32)

    logger.info(
        "Champion model loaded: %s  |  backbone=%s",
        model_path, backbone
    )

    return model, thresholds_np, backbone


# ============================================================
# LOAD CHAMPION METRICS
# ============================================================

def _load_champion_metrics() -> dict:
    """
    Read current champion model card metrics.
    Returns empty dict if no champion exists yet.
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # FIX: read model_card_path directly from registry — no brittle string parsing
    card_path = registry.get("model_card_path", "")

    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})

    return {
        "model_name": card.get("model_name", "unknown"),
        "mean_auc":   float(metrics.get("mean_auc", 0)),
        "mean_f1":    float(metrics.get("mean_f1",  0)),
        "backbone":   card.get("backbone", "unknown"),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_name:        str,
    challenger_mean_auc:    float,
    challenger_val_auc:     float,
    challenger_train_auc:   float,
    challenger_model_path:  str,
    challenger_card_path:   str = "",
) -> dict:
    """
    Compares challenger vs current champion model.

    Args:
        challenger_name:       model name e.g. 'DenseNet121_v20240512'
        challenger_mean_auc:   challenger test mean AUC (14 classes)
        challenger_val_auc:    challenger validation mean AUC
        challenger_train_auc:  challenger training mean AUC
        challenger_model_path: path to challenger .pt file
        challenger_card_path:  path to model card JSON

    Returns:
        result dict with decision: 'PROMOTED' or 'REJECTED'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    # ── If no champion exists → auto-promote ─────────────────
    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_card_path)
        result = {
            "decision":            "PROMOTED",
            "reason":              "No existing champion — first model auto-promoted",
            "challenger_name":     challenger_name,
            "challenger_mean_auc": round(challenger_mean_auc, 4),
            "champion_name":       None,
            "champion_mean_auc":   None,
            "evaluated_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_auc  = champion.get("mean_auc",    0.0)
    champion_name = champion.get("model_name", "unknown")
    gap           = abs(challenger_train_auc - challenger_val_auc)

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-30s  AUC=%.4f", champion_name,     champion_auc)
    logger.info("  Challenger: %-30s  AUC=%.4f", challenger_name, challenger_mean_auc)
    logger.info("  Gap       : %.4f  (train=%.4f  val=%.4f)", gap,
                challenger_train_auc, challenger_val_auc)
    logger.info("=" * 55)

    # ── Promotion gates ───────────────────────────────────────
    gate1_improvement = (challenger_mean_auc - champion_auc) >= MIN_MEAN_AUC_IMPROVEMENT
    gate2_threshold   = challenger_mean_auc >= MIN_MEAN_AUC_THRESHOLD
    gate3_gap         = gap <= MAX_GENERALIZATION_GAP

    gates_passed = gate1_improvement and gate2_threshold and gate3_gap

    if gates_passed:
        decision = "PROMOTED"
        reason   = (
            f"Challenger beats champion: "
            f"mean_AUC {champion_auc:.4f} → {challenger_mean_auc:.4f} "
            f"(+{challenger_mean_auc - champion_auc:.4f})"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_card_path)

    else:
        decision = "REJECTED"
        failed   = []
        if not gate1_improvement:
            failed.append(
                f"AUC improvement {challenger_mean_auc - champion_auc:+.4f} < {MIN_MEAN_AUC_IMPROVEMENT}"
            )
        if not gate2_threshold:
            failed.append(f"mean_AUC {challenger_mean_auc:.4f} < {MIN_MEAN_AUC_THRESHOLD}")
        if not gate3_gap:
            failed.append(f"train-val gap {gap:.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":              decision,
        "reason":                reason,
        "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":       challenger_name,
        "challenger_mean_auc":   round(challenger_mean_auc,   4),
        "challenger_val_auc":    round(challenger_val_auc,    4),
        "challenger_train_auc":  round(challenger_train_auc,  4),
        "challenger_gap":        round(gap,                   4),
        "champion_name":         champion_name,
        "champion_mean_auc":     round(champion_auc,          4),
        "gates": {
            "auc_improvement_passed": gate1_improvement,
            "auc_threshold_passed":   gate2_threshold,
            "gap_passed":             gate3_gap,
        },
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, model_card_path: str = ""):
    # FIX: store model_card_path directly — eliminates brittle filename parsing
    registry = {
        "model_name":      os.path.basename(model_path),
        "model_card_path": model_card_path or "",
    }
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated -> %s", registry["model_name"])


def _save_challenger_log(result: dict):
    """Appends challenger comparison result to challenger_log.json."""
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved -> %s", CHALLENGER_LOG)
