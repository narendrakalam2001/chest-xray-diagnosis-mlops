# ============================================================
# LEAKAGE CHECK — Chest X-Ray Diagnosis ML System
# ============================================================
# Medical imaging leakage is subtler than tabular leakage.
# Key risks:
#   1. Patient-level leakage — same patient in train + test
#   2. Hospital/scanner leakage — systematic institution bias
#   3. Label leakage — using future/derived labels as features
#   4. Preprocessing leakage — fitting normalization on full dataset
# ============================================================

import logging
import numpy as np
import pandas as pd
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

def check_patient_leakage(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    pid_col:  str = "patient_id_inferred",
) -> List[str]:
    """
    CRITICAL: Verify no patient appears in multiple splits.
    Patient leakage inflates AUC by ~5-15% in medical imaging.

    NIH ChestX-ray14 has ~4-6 images per patient.
    Training on patient X, testing on another X-ray of same patient
    = leakage because model learns patient-specific anatomy.
    """
    warnings = []

    # ── Resolve patient ID column ─────────────────────────────
    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        if pid_col not in df.columns and "image_index" in df.columns:
            df[pid_col] = df["image_index"].apply(lambda x: str(x).split("_")[0])

    train_patients = set(df_train.get(pid_col, pd.Series()).dropna())
    val_patients   = set(df_val.get(pid_col,   pd.Series()).dropna())
    test_patients  = set(df_test.get(pid_col,  pd.Series()).dropna())

    # ── Check for overlaps ────────────────────────────────────
    train_val_overlap  = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap   = val_patients   & test_patients

    if train_val_overlap:
        warnings.append(
            f"[LEAKAGE] {len(train_val_overlap)} patients appear in both TRAIN and VAL — patient leakage!"
        )
    if train_test_overlap:
        warnings.append(
            f"[LEAKAGE] {len(train_test_overlap)} patients appear in both TRAIN and TEST — patient leakage!"
        )
    if val_test_overlap:
        warnings.append(
            f"[LEAKAGE] {len(val_test_overlap)} patients appear in both VAL and TEST — patient leakage!"
        )

    if not warnings:
        logger.info(
            "Patient leakage check PASSED  |  train=%d  val=%d  test=%d  (no overlaps)",
            len(train_patients), len(val_patients), len(test_patients)
        )
    else:
        for w in warnings:
            logger.error(w)

    return warnings


# ============================================================
# LABEL LEAKAGE CHECK
# ============================================================

def check_label_leakage(
    metadata_df: pd.DataFrame,
    feature_cols: List[str],
    label_col:   str = "finding_labels",
) -> List[str]:
    """
    Check if any metadata feature is perfectly correlated with labels.
    E.g. if 'follow_up_number' is >0 only when disease is present → leakage.
    """
    warnings = []

    for col in feature_cols:
        try:
            if col not in metadata_df.columns:
                continue

            # Check if binary feature perfectly predicts any label
            has_disease = metadata_df[label_col].apply(
                lambda x: x != "No Finding" and pd.notna(x) and str(x).strip() != ""
            ).astype(int)

            if np.issubdtype(metadata_df[col].dtype, np.number):
                col_binary = (metadata_df[col] > 0).astype(int)
                match_rate = (col_binary == has_disease).mean()
                if match_rate > 0.95:
                    warnings.append(
                        f"[LEAKAGE] Column '{col}' matches disease labels at {match_rate:.2%} rate"
                    )

        except Exception as e:
            logger.debug("Label leakage check failed for '%s': %s", col, e)

    return warnings


# ============================================================
# PREPROCESSING LEAKAGE CHECK
# ============================================================

def check_preprocessing_leakage(
    train_mean: float,
    val_mean:   float,
    test_mean:  float,
    tolerance:  float = 0.05,
) -> List[str]:
    """
    Verify normalization stats were computed ONLY on training set.
    If val/test means match training mean exactly, normalization may
    have been applied to the full dataset (leakage).

    Note: val/test means WILL differ from train mean — that is correct.
    If they are suspiciously close (< tolerance), it may indicate
    the full dataset was used for stat computation.
    """
    warnings = []
    train_val_diff  = abs(train_mean - val_mean)
    train_test_diff = abs(train_mean - test_mean)

    if train_val_diff < tolerance:
        warnings.append(
            f"[WARNING] Train/val pixel mean difference ({train_val_diff:.4f}) < {tolerance} — "
            f"verify normalization stats computed on train only"
        )
    if train_test_diff < tolerance:
        warnings.append(
            f"[WARNING] Train/test pixel mean difference ({train_test_diff:.4f}) < {tolerance} — "
            f"verify normalization stats computed on train only"
        )

    if not warnings:
        logger.info("Preprocessing leakage check PASSED  |  means differ as expected")

    return warnings


# ============================================================
# DUPLICATE IMAGE CHECK
# ============================================================

def check_duplicate_images(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    img_col:  str = "image_index",
) -> List[str]:
    """
    Verify no identical image file appears in both train and test.
    """
    warnings = []

    if img_col not in df_train.columns or img_col not in df_test.columns:
        return warnings

    train_imgs = set(df_train[img_col])
    test_imgs  = set(df_test[img_col])
    overlap    = train_imgs & test_imgs

    if overlap:
        warnings.append(
            f"[LEAKAGE] {len(overlap)} identical images appear in both TRAIN and TEST"
        )
        logger.error("Duplicate image leakage: %d images", len(overlap))
    else:
        logger.info("Duplicate image check PASSED  |  no overlap between train/test images")

    return warnings


# ============================================================
# MASTER LEAKAGE CHECK
# ============================================================

def detect_all_leakage(
    df_train:    pd.DataFrame,
    df_val:      pd.DataFrame,
    df_test:     pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Run all leakage checks. Returns combined list of warnings.
    Call before training — any leakage warning should be investigated.
    """
    all_warnings = []

    # 1. Patient-level leakage
    all_warnings += check_patient_leakage(df_train, df_val, df_test)

    # 2. Duplicate image leakage
    all_warnings += check_duplicate_images(df_train, df_test)

    # 3. Label leakage (optional metadata features)
    if feature_cols:
        all_warnings += check_label_leakage(
            pd.concat([df_train, df_val, df_test], ignore_index=True),
            feature_cols
        )

    if all_warnings:
        logger.error("=" * 55)
        logger.error("LEAKAGE DETECTED — investigate before proceeding")
        for w in all_warnings:
            logger.error("  %s", w)
        logger.error("=" * 55)
    else:
        logger.info("All leakage checks PASSED — safe to proceed with training")

    return all_warnings
