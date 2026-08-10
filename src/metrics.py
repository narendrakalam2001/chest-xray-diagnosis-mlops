# ============================================================
# METRICS — Chest X-Ray Diagnosis ML System
# ============================================================
# Multi-label classification metrics:
#   - ROC-AUC per class + mean AUC (macro)
#   - Per-class F1, Precision, Recall
#   - Threshold tuning (maximize F1 per class)
#   - Clinical metrics (sensitivity, specificity at threshold)
#   - PSI drift monitoring
#   - Uncertainty estimation (prediction entropy)
#   - Radiologist-comparison metrics (CheXNet paper style)
# ============================================================

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
)

from src.config import PATHOLOGY_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)


# ============================================================
# PER-CLASS ROC-AUC
# ============================================================

def compute_per_class_auc(
    y_true:  np.ndarray,
    y_prob:  np.ndarray,
) -> Dict[str, float]:
    """
    Compute ROC-AUC for each of the 14 pathologies.

    Args:
        y_true: (N, 14) ground truth binary labels
        y_prob: (N, 14) predicted probabilities

    Returns:
        Dict mapping label → AUC score.
        Labels with no positive samples get AUC = NaN.
    """
    auc_per_class = {}

    for i, lbl in enumerate(PATHOLOGY_LABELS):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        n_pos = y_t.sum()
        n_neg = (1 - y_t).sum()

        if n_pos == 0 or n_neg == 0:
            auc_per_class[lbl] = float("nan")
            logger.warning("AUC undefined for '%s' (pos=%d, neg=%d)", lbl, n_pos, n_neg)
        else:
            auc_per_class[lbl] = float(roc_auc_score(y_t, y_p))

    return auc_per_class


def compute_mean_auc(auc_dict: Dict[str, float]) -> float:
    """
    Macro mean AUC across all classes (ignoring NaN).
    Primary metric for champion-challenger comparison.
    """
    valid = [v for v in auc_dict.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


# ============================================================
# PER-CLASS AVERAGE PRECISION
# ============================================================

def compute_per_class_ap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Average Precision (area under PR curve) per class.
    More informative than AUC for heavily imbalanced labels.
    """
    ap_per_class = {}
    for i, lbl in enumerate(PATHOLOGY_LABELS):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]
        if y_t.sum() == 0:
            ap_per_class[lbl] = float("nan")
        else:
            ap_per_class[lbl] = float(average_precision_score(y_t, y_p))
    return ap_per_class


# ============================================================
# THRESHOLD TUNING — per class
# ============================================================

def tune_thresholds_per_class(
    y_true:          np.ndarray,
    y_prob:          np.ndarray,
    target_recall:   Optional[float] = None,
    clinical_labels: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Tune decision threshold per class to maximize F1.
    Optionally enforce minimum recall for critical pathologies.

    Clinical override: if clinical_labels dict provided, use those
    fixed thresholds for safety-critical conditions (Pneumonia,
    Pneumothorax) regardless of F1 optimization.

    Args:
        y_true:          (N, 14) ground truth
        y_prob:          (N, 14) probabilities
        target_recall:   if set, find threshold that achieves
                         at least this recall (then maximize precision)
        clinical_labels: dict of {label: threshold} for clinical overrides

    Returns:
        (14,) array of per-class optimal thresholds
    """
    thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

    for i, lbl in enumerate(PATHOLOGY_LABELS):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        # ── Clinical override ─────────────────────────────────
        if clinical_labels and lbl in clinical_labels:
            thresholds[i] = clinical_labels[lbl]
            logger.info("Clinical threshold  %s = %.2f", lbl, thresholds[i])
            continue

        if y_t.sum() == 0:
            continue   # no positives → keep 0.5

        precision, recall, pr_thresholds = precision_recall_curve(y_t, y_p)

        # ── Target recall mode ────────────────────────────────
        if target_recall is not None:
            idxs = np.where(recall >= target_recall)[0]
            if idxs.size > 0:
                best_idx = idxs[np.argmax(precision[idxs])]
                if best_idx < len(pr_thresholds):
                    thresholds[i] = float(pr_thresholds[best_idx])
                    continue

        # ── Maximize F1 ───────────────────────────────────────
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
        best_idx  = np.nanargmax(f1_scores)
        if best_idx < len(pr_thresholds):
            thresholds[i] = float(pr_thresholds[best_idx])

    logger.info("Thresholds tuned  |  range [%.3f, %.3f]",
                thresholds.min(), thresholds.max())
    return thresholds


# ============================================================
# CLASSIFICATION METRICS — per class + summary
# ============================================================

def compute_classification_report(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,   # binary predictions
    y_prob:      np.ndarray,
) -> pd.DataFrame:
    """
    Per-class precision, recall, F1, AUC, AP.
    Returns DataFrame for easy display and logging.
    """
    rows = []

    auc_dict = compute_per_class_auc(y_true, y_prob)
    ap_dict  = compute_per_class_ap(y_true, y_prob)

    for i, lbl in enumerate(PATHOLOGY_LABELS):
        y_t  = y_true[:, i]
        y_pd = y_pred[:, i]
        n_pos = int(y_t.sum())

        if n_pos == 0:
            rows.append({
                "label": lbl, "n_positive": 0,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "roc_auc": np.nan, "avg_precision": np.nan
            })
            continue

        rows.append({
            "label":         lbl,
            "n_positive":    n_pos,
            "precision":     round(float(precision_score(y_t, y_pd, zero_division=0)), 4),
            "recall":        round(float(recall_score(y_t, y_pd, zero_division=0)),    4),
            "f1":            round(float(f1_score(y_t, y_pd, zero_division=0)),         4),
            "roc_auc":       round(float(auc_dict[lbl]) if not np.isnan(auc_dict[lbl]) else np.nan, 4),
            "avg_precision": round(float(ap_dict[lbl])  if not np.isnan(ap_dict[lbl])  else np.nan, 4),
        })

    df = pd.DataFrame(rows)
    logger.info("Per-class metrics computed  |  mean_AUC=%.4f",
                df["roc_auc"].dropna().mean())
    return df


# ============================================================
# CLINICAL METRICS  (sensitivity / specificity)
# ============================================================

def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Clinical sensitivity (TPR) and specificity (TNR) per class.
    Used in radiology AI papers and clinical deployment validation.
    """
    rows = []
    for i, lbl in enumerate(PATHOLOGY_LABELS):
        y_t  = y_true[:, i]
        y_pd = y_pred[:, i]

        if y_t.sum() == 0:
            rows.append({"label": lbl, "sensitivity": np.nan, "specificity": np.nan})
            continue

        tn, fp, fn, tp = confusion_matrix(y_t, y_pd, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)

        rows.append({
            "label":       lbl,
            "TP":          int(tp), "FP": int(fp),
            "TN":          int(tn), "FN": int(fn),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
        })

    return pd.DataFrame(rows)


# ============================================================
# UNCERTAINTY ESTIMATION — prediction entropy
# ============================================================

def compute_prediction_entropy(y_prob: np.ndarray) -> np.ndarray:
    """
    Binary entropy per sample (averaged across 14 classes).
    High entropy → model is uncertain → flag for radiologist review.

    H = - p*log(p) - (1-p)*log(1-p)  per class, then averaged.
    """
    eps = 1e-9
    p   = np.clip(y_prob, eps, 1 - eps)
    H   = -(p * np.log(p) + (1 - p) * np.log(1 - p))  # (N, 14)
    return H.mean(axis=1)   # (N,)


def flag_uncertain_predictions(
    y_prob:    np.ndarray,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    Return boolean mask: True = uncertain = needs radiologist review.
    """
    entropy = compute_prediction_entropy(y_prob)
    return entropy > threshold


# ============================================================
# PSI — Population Stability Index
# ============================================================

def psi(
    expected: np.ndarray,
    actual:   np.ndarray,
    buckets:  int = 10
) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → stable
    PSI 0.1–0.2 → moderate drift (monitor)
    PSI > 0.2   → critical drift (retrain)

    Correct approach:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH using those SAME edges
      3. Compare proportions
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)
        bin_edges  = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


def psi_per_class_score(
    ref_probs: np.ndarray,
    cur_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Compute PSI for each pathology class prediction distribution.
    ref_probs: (N_ref, 14) — training set predictions
    cur_probs: (N_cur, 14) — current batch predictions
    """
    result = {}
    for i, lbl in enumerate(PATHOLOGY_LABELS):
        result[lbl] = psi(ref_probs[:, i], cur_probs[:, i])
    return result


# ============================================================
# COST-SENSITIVE EVALUATION  (clinical risk)
# ============================================================

def clinical_cost_evaluation(
    y_true:          np.ndarray,
    y_pred:          np.ndarray,
    false_neg_cost:  float = 10.0,   # missing disease = high harm
    false_pos_cost:  float = 1.0,    # unnecessary follow-up
) -> Dict:
    """
    Clinical cost model for multi-label X-ray diagnosis.

    False Negative cost >> False Positive cost in medicine:
        FN: missing pneumonia or pneumothorax can be life-threatening
        FP: unnecessary follow-up CT/test = monetary + patient anxiety

    Args:
        false_neg_cost: cost per missed positive (per label averaged)
        false_pos_cost: cost per false alarm

    Returns:
        Dict with total_cost, fn_count, fp_count per critical label
    """
    critical_labels = ["Pneumonia", "Pneumothorax", "Mass", "Nodule"]
    results         = {}
    total_cost      = 0.0

    for lbl in critical_labels:
        i = PATHOLOGY_LABELS.index(lbl)
        y_t  = y_true[:, i]
        y_pd = y_pred[:, i]

        fn = int(((y_t == 1) & (y_pd == 0)).sum())
        fp = int(((y_t == 0) & (y_pd == 1)).sum())

        cost = fn * false_neg_cost + fp * false_pos_cost
        total_cost += cost

        results[lbl] = {
            "false_negatives": fn,
            "false_positives": fp,
            "clinical_cost":   round(cost, 2),
        }

    results["total_clinical_cost"] = round(total_cost, 2)
    logger.info("Clinical cost evaluation  |  total=%.2f", total_cost)
    return results


# ============================================================
# RADIOLOGIST BENCHMARK  (CheXNet-style comparison)
# ============================================================

# Approximate radiologist F1 benchmarks from Rajpurkar et al. 2017
# Used only for display/comparison, not for model decisions
RADIOLOGIST_BENCHMARKS = {
    "Atelectasis":      0.80,
    "Cardiomegaly":     0.90,
    "Effusion":         0.87,
    "Infiltration":     0.69,
    "Mass":             0.78,
    "Nodule":           0.66,
    "Pneumonia":        0.65,
    "Pneumothorax":     0.82,
    "Consolidation":    0.74,
    "Edema":            0.82,
    "Emphysema":        0.87,
    "Fibrosis":         0.80,
    "Pleural_Thickening":0.76,
    "Hernia":           0.91,
}


def compare_to_radiologist(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add radiologist benchmark F1 and delta to metrics DataFrame.
    """
    df = metrics_df.copy()
    df["radiologist_f1"] = df["label"].map(RADIOLOGIST_BENCHMARKS)
    df["vs_radiologist"] = (df["f1"] - df["radiologist_f1"]).round(4)
    return df
