# ============================================================
# PYTEST UNIT TESTS — Chest X-Ray Diagnosis ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Config constants, CLAHETransform, build_train/eval_transform,
#   binarize_predictions, build_threshold_array, denormalize_tensor,
#   parse_finding_labels, labels_to_vector, validate_metadata,
#   compute_label_statistics, compute_pos_weights,
#   patient_level_split, detect_all_leakage,
#   compute_per_class_auc, compute_mean_auc, psi,
#   tune_thresholds_per_class, compute_prediction_entropy,
#   flag_uncertain_predictions, clinical_cost_evaluation,
#   build_model (smoke test), EarlyStopping, MultiLabelHead
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image
import io

from src.config import (
    PATHOLOGY_LABELS, NUM_CLASSES, IMAGE_SIZE,
    PSI_MODERATE, PSI_HIGH,
    MIN_MEAN_AUC_IMPROVEMENT, MIN_MEAN_AUC_THRESHOLD, MAX_GENERALIZATION_GAP,
    CLINICAL_THRESHOLDS, DEFAULT_THRESHOLD, RANDOM_STATE,
)
from src.preprocessing import (
    CLAHETransform, build_train_transform, build_eval_transform,
    binarize_predictions, build_threshold_array, denormalize_tensor,
)
from src.data_loader import (
    parse_finding_labels, labels_to_vector, validate_metadata,
    compute_label_statistics, compute_pos_weights, patient_level_split,
)
from src.leakage_check import (
    check_patient_leakage, check_duplicate_images, detect_all_leakage,
)
from src.metrics import (
    compute_per_class_auc, compute_mean_auc, psi,
    tune_thresholds_per_class, compute_prediction_entropy,
    flag_uncertain_predictions, clinical_cost_evaluation,
    compute_classification_report, psi_per_class_score,
)
from src.model_tuning import (
    build_model, MultiLabelHead, EarlyStopping, BACKBONE_GRID,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def dummy_metadata():
    """Minimal valid metadata DataFrame (50 patients, 200 images)."""
    rows = []
    for pid in range(50):
        for img_n in range(4):
            label = np.random.choice(
                PATHOLOGY_LABELS + ["No Finding", "No Finding", "No Finding"],
            )
            rows.append({
                "image_index":    f"{pid:08d}_{img_n:03d}.png",
                "finding_labels": label,
                "patient_age":    np.random.randint(18, 80),
                "patient_gender": np.random.choice(["M", "F"]),
            })
    df = pd.DataFrame(rows)
    df["patient_id_inferred"] = df["image_index"].apply(lambda x: x.split("_")[0])
    return df


@pytest.fixture
def dummy_probs():
    """(100, 14) random probability array."""
    np.random.seed(42)
    return np.random.rand(100, NUM_CLASSES).astype(np.float32)


@pytest.fixture
def dummy_labels():
    """(100, 14) sparse binary label array."""
    np.random.seed(42)
    labels = np.zeros((100, NUM_CLASSES), dtype=np.float32)
    for i in range(100):
        n_pos = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        idxs  = np.random.choice(NUM_CLASSES, n_pos, replace=False)
        labels[i, idxs] = 1.0
    return labels


@pytest.fixture
def dummy_image_tensor():
    """Normalized (1, 3, 224, 224) image tensor."""
    return torch.randn(1, 3, *IMAGE_SIZE)


@pytest.fixture
def dummy_pil_image():
    """224×224 grayscale PIL image."""
    arr = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


# ============================================================
# CONFIG TESTS  (3 tests)
# ============================================================

class TestConfig:

    def test_num_classes(self):
        """NUM_CLASSES must equal 14 (NIH ChestX-ray14)."""
        assert NUM_CLASSES == 14

    def test_pathology_labels_length(self):
        """PATHOLOGY_LABELS must have exactly 14 entries."""
        assert len(PATHOLOGY_LABELS) == 14

    def test_psi_thresholds_ordered(self):
        """PSI_MODERATE must be less than PSI_HIGH."""
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_positive(self):
        """All champion-challenger thresholds must be positive."""
        assert MIN_MEAN_AUC_IMPROVEMENT > 0
        assert MIN_MEAN_AUC_THRESHOLD > 0
        assert MAX_GENERALIZATION_GAP  > 0

    def test_image_size_tuple(self):
        """IMAGE_SIZE must be a 2-tuple of positive ints."""
        assert len(IMAGE_SIZE) == 2
        assert all(s > 0 for s in IMAGE_SIZE)


# ============================================================
# PREPROCESSING TESTS  (5 tests)
# ============================================================

class TestPreprocessing:

    def test_clahe_output_shape(self, dummy_pil_image):
        """CLAHE output must be same size as input."""
        clahe = CLAHETransform(clip_limit=2.0)
        out   = clahe(dummy_pil_image)
        assert out.size == dummy_pil_image.size

    def test_clahe_output_rgb(self, dummy_pil_image):
        """CLAHE output must be RGB mode."""
        clahe = CLAHETransform()
        out   = clahe(dummy_pil_image)
        assert out.mode == "RGB"

    def test_train_transform_output_shape(self, dummy_pil_image):
        """Train transform must return (3, H, W) tensor."""
        transform = build_train_transform(use_clahe=False)
        tensor    = transform(dummy_pil_image)
        assert tensor.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def test_eval_transform_deterministic(self, dummy_pil_image):
        """Eval transform must be deterministic (same input → same output)."""
        transform = build_eval_transform(use_clahe=False)
        t1 = transform(dummy_pil_image)
        t2 = transform(dummy_pil_image)
        assert torch.allclose(t1, t2), "Eval transform is not deterministic"

    def test_binarize_predictions_default_threshold(self, dummy_probs):
        """binarize_predictions with 0.5 threshold must return 0/1 array."""
        binary = binarize_predictions(dummy_probs)
        assert set(np.unique(binary)).issubset({0, 1})
        assert binary.shape == dummy_probs.shape

    def test_build_threshold_array_length(self):
        """build_threshold_array must return array of length NUM_CLASSES."""
        arr = build_threshold_array(CLINICAL_THRESHOLDS)
        assert len(arr) == NUM_CLASSES

    def test_denormalize_tensor_range(self, dummy_image_tensor):
        """Denormalized tensor values must be in [0, 1]."""
        t    = dummy_image_tensor.squeeze()
        out  = denormalize_tensor(t)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ============================================================
# DATA LOADER TESTS  (6 tests)
# ============================================================

class TestDataLoader:

    def test_parse_finding_labels_no_finding(self):
        """'No Finding' must return empty list."""
        assert parse_finding_labels("No Finding") == []

    def test_parse_finding_labels_single(self):
        """Single label must return list of one element."""
        result = parse_finding_labels("Pneumonia")
        assert result == ["Pneumonia"]

    def test_parse_finding_labels_multi(self):
        """Pipe-separated labels must return correct list."""
        result = parse_finding_labels("Atelectasis|Effusion")
        assert "Atelectasis" in result
        assert "Effusion"    in result
        assert len(result)   == 2

    def test_labels_to_vector_shape(self):
        """labels_to_vector must return array of length 14."""
        vec = labels_to_vector(["Pneumonia", "Effusion"])
        assert vec.shape == (NUM_CLASSES,)

    def test_labels_to_vector_correct_indices(self):
        """Correct indices must be set to 1.0."""
        vec = labels_to_vector(["Pneumonia"])
        pnm_idx = PATHOLOGY_LABELS.index("Pneumonia")
        assert vec[pnm_idx] == 1.0
        assert vec.sum() == 1.0

    def test_labels_to_vector_no_finding(self):
        """Empty label list must return all-zero vector."""
        vec = labels_to_vector([])
        assert vec.sum() == 0.0

    def test_validate_metadata_normalizes_columns(self, dummy_metadata):
        """validate_metadata must not raise on valid input."""
        result = validate_metadata(dummy_metadata)
        assert "image_index"    in result.columns
        assert "finding_labels" in result.columns
        assert len(result) > 0

    def test_compute_pos_weights_shape(self, dummy_metadata):
        """compute_pos_weights must return tensor of length 14."""
        w = compute_pos_weights(dummy_metadata, scale=2.0)
        assert w.shape == (NUM_CLASSES,)
        assert (w > 0).all()

    def test_patient_level_split_no_overlap(self, dummy_metadata):
        """Patient-level split must have NO patient ID overlap across sets."""
        df_train, df_val, df_test = patient_level_split(dummy_metadata)

        pid_col = "patient_id_inferred"
        train_p = set(df_train[pid_col])
        val_p   = set(df_val[pid_col])
        test_p  = set(df_test[pid_col])

        assert len(train_p & val_p)  == 0, "Train/val patient overlap!"
        assert len(train_p & test_p) == 0, "Train/test patient overlap!"
        assert len(val_p   & test_p) == 0, "Val/test patient overlap!"

    def test_patient_level_split_covers_all(self, dummy_metadata):
        """All patients must appear in exactly one split."""
        df_train, df_val, df_test = patient_level_split(dummy_metadata)
        pid_col = "patient_id_inferred"
        all_split = set(df_train[pid_col]) | set(df_val[pid_col]) | set(df_test[pid_col])
        all_orig  = set(dummy_metadata[pid_col])
        assert all_split == all_orig


# ============================================================
# LEAKAGE TESTS  (3 tests)
# ============================================================

class TestLeakage:

    def test_no_leakage_clean_split(self, dummy_metadata):
        """Clean patient-level split must pass leakage check."""
        df_train, df_val, df_test = patient_level_split(dummy_metadata)
        warnings = detect_all_leakage(df_train, df_val, df_test)
        assert len(warnings) == 0, f"Unexpected leakage warnings: {warnings}"

    def test_patient_leakage_detected(self, dummy_metadata):
        """Putting same patients in train+test must trigger leakage warning."""
        df_train = dummy_metadata.iloc[:150].copy()
        df_test  = dummy_metadata.iloc[100:].copy()   # overlap with train
        df_val   = dummy_metadata.iloc[150:175].copy()
        warnings = check_patient_leakage(df_train, df_val, df_test)
        assert len(warnings) > 0, "Patient leakage not detected!"

    def test_duplicate_image_leakage_detected(self, dummy_metadata):
        """Identical images in train+test must trigger leakage warning."""
        df_train = dummy_metadata.iloc[:100].copy()
        df_test  = dummy_metadata.iloc[80:].copy()  # image overlap
        warnings = check_duplicate_images(df_train, df_test)
        assert len(warnings) > 0, "Duplicate image leakage not detected!"


# ============================================================
# METRICS TESTS  (7 tests)
# ============================================================

class TestMetrics:

    def test_per_class_auc_shape(self, dummy_labels, dummy_probs):
        """compute_per_class_auc must return dict with 14 entries."""
        auc_dict = compute_per_class_auc(dummy_labels, dummy_probs)
        valid    = {k: v for k, v in auc_dict.items() if not np.isnan(v)}
        assert len(auc_dict) == NUM_CLASSES
        assert all(0.0 <= v <= 1.0 for v in valid.values())

    def test_mean_auc_range(self, dummy_labels, dummy_probs):
        """Mean AUC must be in [0, 1]."""
        auc_dict = compute_per_class_auc(dummy_labels, dummy_probs)
        mean_auc = compute_mean_auc(auc_dict)
        assert 0.0 <= mean_auc <= 1.0

    def test_psi_stable_identical_distributions(self):
        """PSI of identical distributions must be close to 0."""
        arr = np.random.rand(200)
        p   = psi(arr, arr)
        assert p < 0.05, f"PSI of identical distributions should be ~0, got {p}"

    def test_psi_shifted_distribution(self):
        """PSI of clearly different distributions must exceed PSI_MODERATE."""
        ref = np.random.uniform(0, 0.3, 500)
        cur = np.random.uniform(0.7, 1.0, 500)
        p   = psi(ref, cur)
        assert p > PSI_MODERATE, f"PSI should detect major shift, got {p}"

    def test_tune_thresholds_length(self, dummy_labels, dummy_probs):
        """tune_thresholds_per_class must return array of length 14."""
        thr = tune_thresholds_per_class(dummy_labels, dummy_probs)
        assert len(thr) == NUM_CLASSES

    def test_tune_thresholds_range(self, dummy_labels, dummy_probs):
        """All tuned thresholds must be in (0, 1)."""
        thr = tune_thresholds_per_class(dummy_labels, dummy_probs)
        assert (thr > 0).all() and (thr < 1).all()

    def test_prediction_entropy_shape(self, dummy_probs):
        """Prediction entropy must return (N,) array."""
        H = compute_prediction_entropy(dummy_probs)
        assert H.shape == (dummy_probs.shape[0],)

    def test_prediction_entropy_non_negative(self, dummy_probs):
        """Entropy must be non-negative."""
        H = compute_prediction_entropy(dummy_probs)
        assert (H >= 0).all()

    def test_flag_uncertain_predictions(self, dummy_probs):
        """flag_uncertain_predictions must return boolean array."""
        flags = flag_uncertain_predictions(dummy_probs, threshold=0.1)
        assert flags.dtype == bool
        assert flags.shape == (dummy_probs.shape[0],)

    def test_clinical_cost_evaluation_keys(self, dummy_labels, dummy_probs):
        """clinical_cost_evaluation must return dict with expected keys."""
        y_pred  = binarize_predictions(dummy_probs)
        result  = clinical_cost_evaluation(dummy_labels, y_pred)
        assert "total_clinical_cost" in result
        for lbl in ["Pneumonia", "Pneumothorax", "Mass", "Nodule"]:
            assert lbl in result

    def test_psi_per_class_returns_all_labels(self, dummy_probs):
        """psi_per_class_score must return dict with 14 labels."""
        ref = np.random.rand(200, NUM_CLASSES).astype(np.float32)
        cur = dummy_probs
        result = psi_per_class_score(ref, cur)
        assert len(result) == NUM_CLASSES
        for lbl in PATHOLOGY_LABELS:
            assert lbl in result


# ============================================================
# MODEL TUNING TESTS  (4 tests)
# ============================================================

class TestModelTuning:

    def test_multilabel_head_output_shape(self):
        """MultiLabelHead must output (B, 14) tensor."""
        head = MultiLabelHead(in_features=1024)
        x    = torch.randn(4, 1024)
        out  = head(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_build_model_densenet121_output(self):
        """DenseNet121 model must output (B, 14) on forward pass."""
        model = build_model(backbone="densenet121", pretrained=False, freeze_base=False)
        x     = torch.randn(2, 3, *IMAGE_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_build_model_resnet50_output(self):
        """ResNet50 model must output (B, 14) on forward pass."""
        model = build_model(backbone="resnet50", pretrained=False, freeze_base=False)
        x     = torch.randn(2, 3, *IMAGE_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_backbone_grid_has_required_keys(self):
        """Each entry in BACKBONE_GRID must have required config keys."""
        required = {"backbone", "lr", "weight_decay", "dropout1", "dropout2",
                    "hidden_dim", "unfreeze_n", "scheduler"}
        for name, cfg in BACKBONE_GRID.items():
            missing = required - set(cfg.keys())
            assert not missing, f"BACKBONE_GRID['{name}'] missing: {missing}"

    def test_early_stopping_triggers(self, tmp_path):
        """EarlyStopping must trigger after patience exhausted."""
        stopper  = EarlyStopping(patience=3, checkpoint=str(tmp_path / "ckpt.pt"))
        model    = build_model("densenet121", pretrained=False, freeze_base=False)
        stopped  = False
        for _ in range(10):
            if stopper.step(model, score=0.70):   # never improves
                stopped = True
                break
        assert stopped, "EarlyStopping should have triggered after patience=3"

    def test_early_stopping_saves_checkpoint(self, tmp_path):
        """EarlyStopping must save checkpoint on improvement."""
        ckpt    = str(tmp_path / "best.pt")
        stopper = EarlyStopping(patience=5, checkpoint=ckpt)
        model   = build_model("densenet121", pretrained=False, freeze_base=False)
        stopper.step(model, score=0.80)   # improvement
        assert os.path.exists(ckpt), "Checkpoint was not saved on improvement"


# ============================================================
# SIMULATION TESTS  (2 tests)
# ============================================================

class TestSimulator:

    def test_generate_synthetic_xray_returns_bytes(self):
        """generate_synthetic_xray must return non-empty bytes."""
        from simulation.xray_simulator import generate_synthetic_xray
        img_bytes = generate_synthetic_xray(scenario="mixed", seed=42)
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 100

    def test_synthetic_xray_is_valid_image(self):
        """Generated bytes must be a valid PNG image loadable by PIL."""
        from simulation.xray_simulator import generate_synthetic_xray
        img_bytes = generate_synthetic_xray(scenario="critical", seed=0)
        img       = Image.open(io.BytesIO(img_bytes))
        assert img.size == (224, 224)
        assert img.mode == "RGB"
